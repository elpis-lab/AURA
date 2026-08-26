#include <set>
#include <queue>
#include <limits>
#include <functional>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "ompl/base/goals/GoalRegion.h"
#include "ompl/base/ProblemDefinition.h"
#include "ompl/tools/config/SelfConfig.h"
#include "ompl/base/spaces/SE2StateSpace.h"
#include "ompl/control/planners/aorrt/aorrt.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/base/objectives/MinimaxObjective.h"
#include "ompl/base/objectives/MaximizeMinClearanceObjective.h"
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"
#include "ompl/base/objectives/MechanicalWorkOptimizationObjective.h"

ompl::control::AORRT::AORRT(const SpaceInformationPtr &si) : base::Planner(si, "AORRT")
{
    specs_.approximateSolutions = true;
    siC_ = si.get();
    prevSolution_.clear();
    prevSolutionControls_.clear();
    prevSolutionSteps_.clear();

    Planner::declareParam<double>("goal_bias", this, &AORRT::setGoalBias, &AORRT::getGoalBias, "0.:.05:1.");
    Planner::declareParam<double>("selection_radius", this, &AORRT::setSelectionRadius, &AORRT::getSelectionRadius, "0.:.1:"
                                                                                                                "100");
    Planner::declareParam<bool>("terminate_on_first_solution", this, &AORRT::setTerminateOnFirstSolution, &AORRT::getTerminateOnFirstSolution, "0,1");

}

ompl::control::AORRT::~AORRT()
{
    freeMemory();
}

void ompl::control::AORRT::setup()
{
    base::Planner::setup();
    if (!nn_)
        nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    nn_->setDistanceFunction([this](const Motion *a, const Motion *b)
                             {
                                 return distanceFunction(a, b);
                             });

    if (pdef_)
    {
        if (pdef_->hasOptimizationObjective())
        {
            opt_ = pdef_->getOptimizationObjective();
            if (dynamic_cast<base::MaximizeMinClearanceObjective *>(opt_.get()) ||
                dynamic_cast<base::MinimaxObjective *>(opt_.get()))
                OMPL_WARN("%s: Asymptotic near-optimality has only been proven with Lipschitz continuous cost "
                          "functions w.r.t. state and control. This optimization objective will result in undefined "
                          "behavior",
                          getName().c_str());
        }
        else
        {
            OMPL_WARN("%s: No optimization object set. Using path length", getName().c_str());
            opt_ = std::make_shared<base::PathLengthOptimizationObjective>(si_);
            pdef_->setOptimizationObjective(opt_);
        }
    }

    prevSolutionCost_ = opt_->infiniteCost();
}



void ompl::control::AORRT::clear()
{
    Planner::clear();
    sampler_.reset();
    controlSampler_.reset();
    freeMemory();
    if (nn_)
        nn_->clear();
    if (opt_)
        prevSolutionCost_ = opt_->infiniteCost();
    
    // Clear all stored solutions
    allSolutions_.clear();

}

void ompl::control::AORRT::freeMemory()
{
    if (nn_)
    {
        std::vector<Motion *> motions;
        nn_->list(motions);
        for (auto &motion : motions)
        {
            if (motion->state_)
                si_->freeState(motion->state_);
            if (motion->control_)
                siC_->freeControl(motion->control_);
            delete motion;
        }
    }
    
    for (auto &i : prevSolution_)
    {
        if (i)
            si_->freeState(i);
    }
    prevSolution_.clear();
    for (auto &prevSolutionControl : prevSolutionControls_)
    {
        if (prevSolutionControl)
            siC_->freeControl(prevSolutionControl);
    }
    prevSolutionControls_.clear();
    prevSolutionSteps_.clear();
}

ompl::control::AORRT::Motion *ompl::control::AORRT::selectNode(ompl::control::AORRT::Motion *sample)
{
    std::vector<Motion *> ret;
    Motion *selected = nullptr;
    base::Cost bestCost = opt_->infiniteCost();
    
    // Track selection statistics (commented out for production)
    // static unsigned totalSelections = 0;
    // static unsigned radiusSelections = 0;
    // static unsigned fallbackSelections = 0;
    
    nn_->nearestR(sample, selectionRadius_, ret);
    for (auto &i : ret)
    {
        if (!i->inactive_ && opt_->isCostBetterThan(i->accCost_, bestCost))
        {
            bestCost = i->accCost_;
            selected = i;
        }
    }
    
    if (selected == nullptr)
    {
        // Fallback to nearest neighbor
        int k = 1;
        while (selected == nullptr)
        {
            nn_->nearestK(sample, k, ret);
            for (unsigned int i = 0; i < ret.size() && selected == nullptr; i++)
                if (!ret[i]->inactive_)
                    selected = ret[i];
            k += 5;
        }
        // fallbackSelections++;
    }
    else
    {
        // radiusSelections++;
    }
    
    // totalSelections++;
    
    // Output selection statistics every 1000 selections (commented out for production)
    // if (totalSelections % 1000 == 0)
    // {
    //     OMPL_INFORM("Selection stats: radius=%.1f%%, fallback=%.1f%% (radius=%.3f)", 
    //                 (double)radiusSelections/totalSelections*100.0,
    //                 (double)fallbackSelections/totalSelections*100.0,
    //                 selectionRadius_);
    // }
    
    return selected;
}


ompl::base::PlannerStatus ompl::control::AORRT::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    base::Goal *goal = pdef_->getGoal().get();
    auto *goal_s = dynamic_cast<base::GoalSampleableRegion *>(goal);

    // Initialize convergence tracking (commented out for production)
    // static std::vector<std::pair<unsigned, double>> convergenceHistory;
    // static unsigned lastConvergenceOutput = 0;
    // static auto lastTime = std::chrono::high_resolution_clock::now();

    while (const base::State *st = pis_.nextStart())
    {
        auto *motion = new Motion(siC_);
        si_->copyState(motion->state_, st);
        siC_->nullControl(motion->control_);
        nn_->add(motion);
        motion->accCost_ = opt_->identityCost();
    }

    if (nn_->size() == 0)
    {
        OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
        return base::PlannerStatus::INVALID_START;
    }

    if (!sampler_)
        sampler_ = si_->allocStateSampler();
    if (!controlSampler_)
        controlSampler_ = siC_->allocControlSampler();

    const base::ReportIntermediateSolutionFn intermediateSolutionCallback = pdef_->getIntermediateSolutionCallback();

    OMPL_INFORM("%s: Starting planning with %u states already in datastructure\n", getName().c_str(), nn_->size());

    Motion *solution = nullptr;
    Motion *approxsol = nullptr;
    double approxdif = std::numeric_limits<double>::infinity();
    bool sufficientlyShort = false;

    auto *rmotion = new Motion(siC_);
    base::State *rstate = rmotion->state_;
    Control *rctrl = rmotion->control_;
    base::State *xstate = si_->allocState();

    unsigned iterations = 0;
    
    // Change the condition  based on replan flag
    while (ptc == false)
    {
        /* sample random state (with goal biasing) */
        if (goal_s && rng_.uniform01() < goalBias_ && goal_s->canSample())
            goal_s->sampleGoal(rstate);
        else
            sampler_->sampleUniform(rstate);

        /* find closest state in the tree */
        Motion *nmotion = selectNode(rmotion);

        /* sample a random control that attempts to go towards the random state, and also sample a control duration */
        controlSampler_->sample(rctrl);
        unsigned int cd = rng_.uniformInt(siC_->getMinControlDuration(), siC_->getMaxControlDuration());
        unsigned int propCd = siC_->propagateWhileValid(nmotion->state_, rctrl, cd, rstate);

        if (propCd == cd)
        {
            base::Cost incCostMotion = opt_->motionCost(nmotion->state_, rstate);
            base::Cost incCostControl = opt_->controlCost(rctrl, cd);
            base::Cost incCost = opt_->combineCosts(incCostMotion, incCostControl);
            base::Cost cost = opt_->combineCosts(nmotion->accCost_, incCost);

            if (opt_->isCostBetterThan(cost, bestSolutionCost_))
            {
                /* create a motion */
                auto *motion = new Motion(siC_);
                motion->accCost_ = cost;
                si_->copyState(motion->state_, rmotion->state_);
                siC_->copyControl(motion->control_, rctrl);
                motion->steps_ = cd;
                motion->parent_ = nmotion;
                nmotion->children_.push_back(motion);
                nmotion->numChildren_++;

                nn_->add(motion);
                double dist = 0.0;
                bool solv = goal->isSatisfied(motion->state_, &dist);
                            
                
                if (solv)
                {
                    approxdif = dist;
                    solution = motion;

                    // Add cost improvement analysis (keep this - it's essential)
                    double costImprovement = 0.0;
                    if (prevSolutionCost_.value() != std::numeric_limits<double>::infinity())
                    {
                        costImprovement = prevSolutionCost_.value() - solution->accCost_.value();
                        OMPL_INFORM("Found solution with cost %.4f (improvement: %.4f, %.1f%% better)", 
                                    solution->accCost_.value(), costImprovement, 
                                    (costImprovement / prevSolutionCost_.value()) * 100.0);
                    }
                    else
                    {
                        OMPL_INFORM("Found FIRST solution with cost %.4f", solution->accCost_.value());
                    }

                    for (auto &i : prevSolution_)
                        if (i)
                            si_->freeState(i);
                    prevSolution_.clear();
                    for (auto &prevSolutionControl : prevSolutionControls_)
                        if (prevSolutionControl)
                            siC_->freeControl(prevSolutionControl);
                    prevSolutionControls_.clear();
                    prevSolutionSteps_.clear();

                    Motion *solTrav = solution;
                    while (solTrav->parent_ != nullptr)
                    {
                        prevSolution_.push_back(si_->cloneState(solTrav->state_));
                        prevSolutionControls_.push_back(siC_->cloneControl(solTrav->control_));
                        prevSolutionSteps_.push_back(solTrav->steps_);
                        solTrav = solTrav->parent_;
                    }
                    prevSolution_.push_back(si_->cloneState(solTrav->state_));
                    prevSolutionCost_ = solution->accCost_;

                    // Track convergence history (commented out for production)
                    // convergenceHistory.push_back({iterations, solution->accCost_.value()});
                    
                    // Output convergence analysis every 5 solutions (commented out for production)
                    // if (convergenceHistory.size() - lastConvergenceOutput >= 5)
                    // {
                    //     OMPL_INFORM("Convergence analysis (last 5 solutions):");
                    //     size_t start = convergenceHistory.size() - 5;
                    //     for (size_t i = start; i < convergenceHistory.size(); ++i)
                    //     {
                    //         double improvement = (i > start) ? 
                    //             convergenceHistory[i-1].second - convergenceHistory[i].second : 0.0;
                    //         OMPL_INFORM("  Iter %u: cost=%.4f (improvement: %.4f)", 
                    //                     convergenceHistory[i].first, convergenceHistory[i].second, improvement);
                    //         }
                    //     lastConvergenceOutput = convergenceHistory.size();
                    // }

                    bestSolutionCost_ = solution->accCost_;
                    bestSolutionPath_ = std::make_shared<PathControl>(si_);
                    for (int i = prevSolution_.size() - 1; i >= 1; --i)
                        bestSolutionPath_->append(prevSolution_[i], prevSolutionControls_[i - 1],
                                    prevSolutionSteps_[i - 1] * siC_->getPropagationStepSize());
                    bestSolutionPath_->append(prevSolution_[0]);

                    auto path(std::make_shared<PathControl>(si_));
                    for (int i = prevSolution_.size() - 1; i >= 1; --i)
                        path->append(prevSolution_[i], prevSolutionControls_[i - 1],
                                    prevSolutionSteps_[i - 1] * siC_->getPropagationStepSize());
                    path->append(prevSolution_[0]);
                    ompl::base::PlannerSolution path2solution(path);
                    path2solution.cost_ = solution->accCost_;
                    pdef_->addSolutionPath(path2solution);

                    // Store the solution in our vector for tracking
                    allSolutions_.push_back(path2solution);

                    if (intermediateSolutionCallback)
                    {
                        // the callback requires a vector with const elements -> create a copy
                        std::vector<const base::State *> prevSolutionConst(prevSolution_.begin(), prevSolution_.end());
                        intermediateSolutionCallback(this, prevSolutionConst, prevSolutionCost_);
                    }
                    sufficientlyShort = opt_->isSatisfied(solution->accCost_);
                    if (sufficientlyShort)
                        break;
                    if (opt_->isCostBetterThan(motion->accCost_, bestSolutionCost_))
                    {
                        bestSolutionCost_ = motion->accCost_;
                        // Re-create the class member to a new, empty path
                        bestSolutionPath_ = std::make_shared<PathControl>(si_);
                        for (int i = prevSolution_.size() - 2; i >= 1; --i)
                            bestSolutionPath_->append(prevSolution_[i], prevSolutionControls_[i - 1],
                                        prevSolutionSteps_[i - 1] * siC_->getPropagationStepSize());
                        bestSolutionPath_->append(prevSolution_[0]);
                    }
                }
                if (solution == nullptr && dist < approxdif)
                {
                    OMPL_INFORM("Found APPROXIMATE solution with cost %.4f and distance %.4f", motion->accCost_.value(), dist);
                    approxdif = dist;
                    approxsol = motion;

                    for (auto &i : prevSolution_)
                        if (i)
                            si_->freeState(i);
                    prevSolution_.clear();
                    for (auto &prevSolutionControl : prevSolutionControls_)
                        if (prevSolutionControl)
                            siC_->freeControl(prevSolutionControl);
                    prevSolutionControls_.clear();
                    prevSolutionSteps_.clear();

                    Motion *solTrav = approxsol;
                    while (solTrav->parent_ != nullptr)
                    {
                        prevSolution_.push_back(si_->cloneState(solTrav->state_));
                        prevSolutionControls_.push_back(siC_->cloneControl(solTrav->control_));
                        prevSolutionSteps_.push_back(solTrav->steps_);
                        solTrav = solTrav->parent_;
                    }
                    prevSolution_.push_back(si_->cloneState(solTrav->state_));


                }

                if (nmotion != rmotion)
                {
                    while (nmotion->inactive_ && nmotion->numChildren_ == 0)
                    {
                        nmotion->inactive_ = true;
                        nn_->remove(nmotion);

                        if (nmotion->state_)
                            si_->freeState(nmotion->state_);
                        if (nmotion->control_)
                            siC_->freeControl(nmotion->control_);

                        nmotion->state_ = nullptr;
                        nmotion->control_ = nullptr;
                        nmotion->parent_->numChildren_--;
                        nmotion->parent_->children_.erase(std::remove(nmotion->parent_->children_.begin(), nmotion->parent_->children_.end(), nmotion), nmotion->parent_->children_.end());
                        Motion *nmotionParent = nmotion->parent_;
                        delete nmotion;
                        nmotion = nmotionParent;
                    }
                }
            }
        }
        
        iterations++;
        
        // Periodic progress and tree statistics (commented out for production - too verbose)
        // if (iterations % 10000 == 0)
        // {
        //     OMPL_INFORM("AORRT Progress: %u iterations, %u states, best cost: %.4f", 
        //                 iterations, nn_->size(), bestSolutionCost_.value());
        //     
        //     // Tree depth analysis
        //     unsigned maxDepth = 0;
        //     unsigned totalDepth = 0;
        //     unsigned leafCount = 0;
        //     
        //     std::vector<Motion*> allMotions;
        //     nn_->list(allMotions);
        //     
        //     for (auto* motion : allMotions)
        //     {
        //         if (motion->numChildren_ == 0) // Leaf node
        //         {
        //             leafCount++;
        //             unsigned depth = 0;
        //             Motion* current = motion;
        //             while (current->parent_)
        //             {
        //                 depth++;
        //                 current = current->parent_;
        //             }
        //             totalDepth += depth;
        //             maxDepth = std::max(maxDepth, depth);
        //         }
        //     }
        //     
        //     if (leafCount > 0)
        //     {
        //         double avgDepth = (double)totalDepth / leafCount;
        //         OMPL_INFORM("  Tree stats: max depth=%u, avg depth=%.1f, leaf nodes=%u", 
        //                     maxDepth, avgDepth, leafCount);
        //     }
        // }
        
        // Performance monitoring (commented out for production)
        // if (iterations % 50000 == 0)
        // {
        //     // Calculate states per second
        //     auto currentTime = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);
        //     double statesPerSecond = 50000.0 / (duration.count() / 1000.0);
        //     
        //     OMPL_INFORM("Performance: %.1f states/sec, %.1f iterations/sec", 
        //                 statesPerSecond, statesPerSecond * (double)iterations / nn_->size());
        //     
        //     lastTime = currentTime;
        // }
    }

    bool solved = false;
    bool approximate = false;
    if (solution == nullptr)
    {
        solution = approxsol;
        approximate = true;
    }

    if (solution != nullptr)
    {
        solved = true;
        
        // Calculate solution path statistics (keep this - it's useful)
        unsigned pathLength = prevSolution_.size();
        double totalPathCost = solution->accCost_.value();
        double avgCostPerStep = totalPathCost / (pathLength - 1);
        
        OMPL_INFORM("Solution path: %u states, %.4f total cost, %.4f avg cost/step", 
                    pathLength, totalPathCost, avgCostPerStep);
        
        // Show cost breakdown if using mechanical work objective (keep this)
        if (dynamic_cast<base::MechanicalWorkOptimizationObjective*>(opt_.get()))
        {
            OMPL_INFORM("  Mechanical work objective: path length + control effort");
        }
        else if (dynamic_cast<base::PathLengthOptimizationObjective*>(opt_.get()))
        {
            OMPL_INFORM("  Path length objective: Euclidean distance optimization");
        }
    }

    si_->freeState(xstate);
    if (rmotion->state_)
        si_->freeState(rmotion->state_);
    if (rmotion->control_)
        siC_->freeControl(rmotion->control_);
    delete rmotion;

    OMPL_INFORM("%s: Created %u states in %u iterations", getName().c_str(), nn_->size(), iterations);

    // Enhanced final statistics (keep this - it's essential summary)
    if (solved)
    {
        OMPL_INFORM("Final solution statistics:");
        OMPL_INFORM("  Best solution cost: %.4f", bestSolutionCost_.value());
        OMPL_INFORM("  Total solutions found: %zu", allSolutions_.size());
        
        if (allSolutions_.size() > 1)
        {
            double firstCost = allSolutions_[0].cost_.value();
            double finalCost = allSolutions_.back().cost_.value();
            double totalImprovement = firstCost - finalCost;
            double improvementPercentage = (totalImprovement / firstCost) * 100.0;
            
            OMPL_INFORM("  Cost improvement: %.4f (%.1f%%)", totalImprovement, improvementPercentage);
            OMPL_INFORM("  Average cost improvement per solution: %.4f", 
                        totalImprovement / (allSolutions_.size() - 1));
        }
        
        // Calculate max tree depth for final stats (keep this - useful summary)
        unsigned maxTreeDepth = 0;
        std::vector<Motion*> allMotions;
        nn_->list(allMotions);
        
        for (auto* motion : allMotions)
        {
            if (motion->numChildren_ == 0) // Leaf node
            {
                unsigned depth = 0;
                Motion* current = motion;
                while (current->parent_)
                {
                    depth++;
                    current = current->parent_;
                }
                maxTreeDepth = std::max(maxTreeDepth, depth);
            }
        }
        
        OMPL_INFORM("  Tree depth: max=%u, states=%u", maxTreeDepth, nn_->size());
    }

    return {solved, approximate};
}


ompl::base::PlannerStatus ompl::control::AORRT::resolve(const double replanning_time)
{
    checkValidity();
    const auto resolveStarted = std::chrono::steady_clock::now();
    OMPL_INFORM("Starting: replan function");
    
    // 1. Get the best solution from allSolutions_ before clearing
    OMPL_INFORM("Getting best solution from allSolutions_ (has %d solutions)", (int)allSolutions_.size());
    if (allSolutions_.empty()) {
        OMPL_WARN("%s: No solutions in allSolutions_ for replan.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    // Find the best solution (lowest cost) from allSolutions_
    auto bestSolutionIt = std::min_element(allSolutions_.begin(), allSolutions_.end(),
        [](const ompl::base::PlannerSolution& a, const ompl::base::PlannerSolution& b) {
            return a.cost_.value() < b.cost_.value();
        });
    const base::Cost selectedSolutionCost = bestSolutionIt->cost_;
    
    auto pathControl = std::dynamic_pointer_cast<PathControl>(bestSolutionIt->path_);
    OMPL_INFORM("Using best solution from allSolutions_ with cost %.4f and %d states", 
                selectedSolutionCost.value(),
                pathControl ? pathControl->getStateCount() : 0);
    
    // Clear all previous solutions after getting the best one
    OMPL_INFORM("Clearing all previous solutions from allSolutions_");
    allSolutions_.clear();

    if (!pathControl || pathControl->getStateCount() < 2)
    {
        OMPL_WARN("%s: Best solution from allSolutions_ has less than 2 states, cannot replan.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    ompl::base::State *newState = pathControl->getState(1);
    OMPL_INFORM("Replan: Starting from second state of best solution from allSolutions_");
    
    // 2. Find the motion corresponding to this state (newStart)
    OMPL_INFORM("Starting: Find the motion corresponding to the new state");
    Motion *tempMotionForSearch = new Motion(siC_);
    si_->copyState(tempMotionForSearch->state_, newState);
    Motion *newStart = nn_->nearest(tempMotionForSearch);
    si_->freeState(tempMotionForSearch->state_);
    siC_->freeControl(tempMotionForSearch->control_);
    delete tempMotionForSearch;
    OMPL_INFORM("Finished: Find the motion corresponding to the new state");
    
    if (!newStart)
    {
        OMPL_WARN("%s: Could not find motion corresponding to new state.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    double distance = si_->distance(newStart->state_, newState);
    if (distance > 1e-5)
    {
        OMPL_WARN("%s: Could not find exact motion for new state (distance=%.8f).", getName().c_str(), distance);
        return base::PlannerStatus::ABORT;
    }
    
    OMPL_INFORM("Replan: Found motion corresponding to new state");
    
    // 3. Set toKeep to false for all motions initially
    OMPL_INFORM("Starting: Set toKeep to false for all motions");
    std::vector<Motion*> allMotions;
    nn_->list(allMotions);
    for (Motion* motion : allMotions)
    {
        motion->toKeep_ = false;
    }
    OMPL_INFORM("Finished: Set toKeep to false for all motions");
    
    // 4. Set toKeep to true for the branch starting from the second motion in solution path
    OMPL_INFORM("Starting: Marking motions to keep (branch from second state)");
    std::queue<Motion*> branchQueue;
    branchQueue.push(newStart);
    newStart->toKeep_ = true;
    int keepCount = 1; // Start with 1 (the newStart motion)
    
    // Process the branch in breadth-first order
    while (!branchQueue.empty())
    {
        Motion* current = branchQueue.front();
        branchQueue.pop();
        
        // Add all children to the queue and mark them to keep
        for (Motion* child : current->children_)
        {
            child->toKeep_ = true;
            keepCount++;
            branchQueue.push(child);
        }
    }
    
    // Additionally, mark all motions that are part of the original solution path
    // This ensures the original solution remains available
    // //////////////////////////////////////////////////////////////////////////
    // OMPL_INFORM("Starting: Marking motions in original solution path");
    // std::set<Motion*> originalPathMotions;
    
    // // Find all motions that correspond to states in the original solution path
    // for (size_t i = 1; i < pathControl->getStateCount(); ++i) {
    //     ompl::base::State* pathState = pathControl->getState(i);
        
    //     // Find the motion in the tree that corresponds to this state
    //     Motion* tempMotion = new Motion(siC_);
    //     si_->copyState(tempMotion->state_, pathState);
    //     Motion* correspondingMotion = nn_->nearest(tempMotion);
    //     delete tempMotion;
        
    //     if (correspondingMotion) {
    //         double dist = si_->distance(correspondingMotion->state_, pathState);
    //         if (dist < 1e-5) { // Close enough to be the same state
    //             originalPathMotions.insert(correspondingMotion);
    //             correspondingMotion->toKeep_ = true;
    //             keepCount++;
                
    //             // Also mark all ancestors of this motion to preserve the path
    //             Motion* ancestor = correspondingMotion->parent_;
    //             while (ancestor && !ancestor->toKeep_) {
    //                 ancestor->toKeep_ = true;
    //                 keepCount++;
    //                 ancestor = ancestor->parent_;
    //             }
    //         }
    //     }
    // }
    // OMPL_INFORM("Finished: Marking motions in original solution path");
    
    OMPL_INFORM("Finished: Marking motions to keep (branch from second state)");
    
    OMPL_INFORM("Replan: Marked %d motions to keep (including original solution path)", keepCount);
    
    // 6. Rebuild the nearest-neighbor structure once. Repeated nn_->remove()
    // is quadratic for the large trees produced by an initial AORRT solve and
    // used to turn a nominal real-time replan into a minute-long operation.
    OMPL_INFORM("Starting: Rebuild tree with motions marked to keep");
    std::vector<Motion*> keptMotions;
    std::vector<Motion*> removedMotions;
    keptMotions.reserve(static_cast<std::size_t>(keepCount));
    removedMotions.reserve(allMotions.size() - static_cast<std::size_t>(keepCount));
    for (Motion* motion : allMotions)
    {
        (motion->toKeep_ ? keptMotions : removedMotions).push_back(motion);
    }
    nn_->clear();
    if (!keptMotions.empty())
        nn_->add(keptMotions);
    for (Motion* motion : removedMotions)
    {
        if (motion->state_)
            si_->freeState(motion->state_);
        if (motion->control_)
            siC_->freeControl(motion->control_);
        delete motion;
    }
    const int removedCount = static_cast<int>(removedMotions.size());
    OMPL_INFORM("Finished: Rebuild tree with motions marked to keep");
    
    // Debug: Check if all states from bestSolutionPath_ are still in the tree
    if (bestSolutionPath_ && bestSolutionPath_->getStateCount() > 0) {
        OMPL_INFORM("Debug: Checking if all states from bestSolutionPath_ are still in the tree after pruning");
        for (size_t i = 0; i < bestSolutionPath_->getStateCount(); ++i) {
            ompl::base::State* state = bestSolutionPath_->getState(i);
            Motion* tempMotion = new Motion(siC_);
            si_->copyState(tempMotion->state_, state);
            Motion* found = nn_->nearest(tempMotion);
            double dist = si_->distance(found->state_, state);
            if (dist < 1e-5) {
                OMPL_INFORM("  State %zu: PRESENT in tree (distance=%.8f)", i, dist);
            } else {
                OMPL_WARN("  State %zu: MISSING from tree (nearest distance=%.8f)", i, dist);
            }
            si_->freeState(tempMotion->state_);
            siC_->freeControl(tempMotion->control_);
            delete tempMotion;
        }
    }

    OMPL_INFORM("Replan: Removed %d motions, kept %d motions", removedCount, keepCount);
    

    
    // 8. Make newStart the root (no parent)
    OMPL_INFORM("Starting: Make newStart the root");
    newStart->parent_ = nullptr;
    
    // 9. Recalculate accumulated costs relative to the new root. Keeping the
    // removed prefix in accCost_ makes all post-resolve objective comparisons
    // incorrect, particularly when edge durations differ.
    std::queue<Motion*> costQueue;
    costQueue.push(newStart);
    newStart->accCost_ = opt_->identityCost();
    while (!costQueue.empty())
    {
        Motion* current = costQueue.front();
        costQueue.pop();
        for (Motion* child : current->children_)
        {
            const base::Cost incCostMotion =
                opt_->motionCost(current->state_, child->state_);
            const base::Cost incCostControl =
                opt_->controlCost(child->control_, child->steps_);
            child->accCost_ = opt_->combineCosts(
                current->accCost_,
                opt_->combineCosts(incCostMotion, incCostControl));
            costQueue.push(child);
        }
    }
    
    
    OMPL_INFORM("Replan: Rebuilt tree with %d motions, new start cost: %.4f", 
                nn_->size(), newStart->accCost_.value());
    
    // 11. Set the start state in the problem definition to newState
    OMPL_INFORM("Starting: Set the start state in the problem definition");
    pdef_->clearStartStates();
    pdef_->addStartState(newState);
    OMPL_INFORM("Finished: Set the start state in the problem definition");
    
    // Create a shortened path from the current solution by removing the first state/control
    OMPL_INFORM("Creating shortened path by removing first state/control from current solution");
    auto shortenedPath = std::make_shared<PathControl>(si_);
    
    // Add all states and controls starting from index 1 (skip the first state/control)
    for (size_t i = 1; i < pathControl->getStateCount(); ++i)
    {
        if (i == 1)
        {
            // First state in shortened path (was second state in original path)
            shortenedPath->append(pathControl->getState(i));
        }
        else if (i < pathControl->getStateCount())
        {
            // Add control and state for remaining segments
            shortenedPath->append(pathControl->getState(i), 
                               pathControl->getControl(i-1),
                               pathControl->getControlDuration(i-1));
        }
    }
    
    OMPL_INFORM("Shortened path created with %d states and %d controls", 
                shortenedPath->getStateCount(), shortenedPath->getControlCount());
    
    base::Cost shortenedCost = opt_->identityCost();
    for (unsigned int i = 0; i < shortenedPath->getControlCount(); ++i)
    {
        const auto steps = static_cast<unsigned int>(std::lround(
            shortenedPath->getControlDuration(i) /
            siC_->getPropagationStepSize()));
        const base::Cost edgeCost = opt_->combineCosts(
            opt_->motionCost(
                shortenedPath->getState(i),
                shortenedPath->getState(i + 1)),
            opt_->controlCost(shortenedPath->getControl(i), steps));
        shortenedCost = opt_->combineCosts(shortenedCost, edgeCost);
    }
    prevSolutionCost_ = shortenedCost;
    bestSolutionCost_ = shortenedCost;
    
    // Clear existing solution paths and add the shortened path as a PlannerSolution
    OMPL_INFORM("Clearing all existing solution paths before solve");
    pdef_->clearSolutionPaths();
    OMPL_INFORM("Creating PlannerSolution for shortened path");
    ompl::base::PlannerSolution path2solution(shortenedPath);
    OMPL_INFORM("Setting cost of shortened path to %.4f", shortenedCost.value());
    path2solution.cost_ = shortenedCost;
    OMPL_INFORM("Adding shortened path as PlannerSolution to the problem definition");
    pdef_->addSolutionPath(path2solution);
    OMPL_INFORM("Finished: Add shortened path as PlannerSolution");
    
    // Store the solution in our vector for tracking
    allSolutions_.push_back(path2solution);
    
    OMPL_INFORM("Replan: shortened path - removed first state/control, now has %d states and %d controls", 
                shortenedPath->getStateCount(), shortenedPath->getControlCount());

    // 13. Run the solve function to continue planning from the new start state
    OMPL_INFORM("Starting: Run solve to continue planning from new start state");
    const double preparationSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - resolveStarted).count();
    const double remainingSeconds = std::max(0.0, replanning_time - preparationSeconds);
    base::PlannerStatus status = base::PlannerStatus::EXACT_SOLUTION;
    if (remainingSeconds > 0.0)
    {
        base::PlannerTerminationCondition ptc =
            base::timedPlannerTerminationCondition(remainingSeconds);
        status = solve(ptc);
    }
    else
        OMPL_WARN("%s: Tree maintenance consumed the %.6f s replanning budget; retaining the shortened exact path.",
                  getName().c_str(), replanning_time);
    OMPL_INFORM("Finished: Run solve to continue planning from new start state");
    
    OMPL_INFORM("Replan: solve() completed with status: %s", status ? "SUCCESS" : "FAILED");
    OMPL_INFORM("Finished: replan function");
    return status;
}

void ompl::control::AORRT::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<Motion *> motions;
    std::vector<Motion *> allMotions;
    if (nn_)
        nn_->list(motions);

    for (auto &motion : motions)
    {
        if (motion->numChildren_ == 0)
        {
            allMotions.push_back(motion);
        }
    }
    for (unsigned i = 0; i < allMotions.size(); i++)
    {
        if (allMotions[i]->parent_ != nullptr)
        {
            allMotions.push_back(allMotions[i]->parent_);
        }
    }

    double delta = siC_->getPropagationStepSize();

    if (prevSolution_.size() != 0)
        data.addGoalVertex(base::PlannerDataVertex(prevSolution_[0]));

    for (auto m : allMotions)
    {
        if (m->parent_)
        {
            if (data.hasControls())
                data.addEdge(base::PlannerDataVertex(m->parent_->state_), base::PlannerDataVertex(m->state_),
                             control::PlannerDataEdgeControl(m->control_, m->steps_ * delta));
            else
                data.addEdge(base::PlannerDataVertex(m->parent_->state_), base::PlannerDataVertex(m->state_));
        }
        else
            data.addStartVertex(base::PlannerDataVertex(m->state_));
    }
}

void ompl::control::AORRT::costTrackingThread(const std::string& filename, 
                                               std::chrono::time_point<std::chrono::system_clock> startTime) const
{
    // Dummy implementation for Python binding compatibility
    // This function was removed but Python bindings still expect it
    // Do nothing - cost tracking functionality has been removed
}

const std::vector<ompl::base::PlannerSolution>& ompl::control::AORRT::getAllSolutions() const
{
    return allSolutions_;
}

void ompl::control::AORRT::clearAllSolutions()
{
    allSolutions_.clear();
}
