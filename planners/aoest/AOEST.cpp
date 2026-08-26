/*
 * Author: Ali Golestaneh
 */

#include <set>
#include <queue>
#include <limits>
#include <functional>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <unordered_map>   // === AOEST: added (not strictly required, but handy if you extend) ===
#include <sstream>         // === AOEST: added (ditto) ===
#include "ompl/base/goals/GoalRegion.h"
#include "ompl/base/ProblemDefinition.h"
#include "ompl/tools/config/SelfConfig.h"
#include "ompl/base/spaces/SE2StateSpace.h"
#include "ompl/control/planners/aoest/AOEST.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/base/objectives/MinimaxObjective.h"
#include "ompl/base/objectives/MaximizeMinClearanceObjective.h"
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"
#include "ompl/base/objectives/MechanicalWorkOptimizationObjective.h"

ompl::control::AOEST::AOEST(const SpaceInformationPtr &si) : base::Planner(si, "AOEST")
{
    specs_.approximateSolutions = true;
    siC_ = si.get();
    prevSolution_.clear();
    prevSolutionControls_.clear();
    prevSolutionSteps_.clear();

    Planner::declareParam<double>("goal_bias", this, &AOEST::setGoalBias, &AOEST::getGoalBias, "0.:.05:1.");
    Planner::declareParam<double>("selection_radius", this, &AOEST::setSelectionRadius, &AOEST::getSelectionRadius, "0.:.1:"
                                                                                                                "100");
    Planner::declareParam<bool>("terminate_on_first_solution", this, &AOEST::setTerminateOnFirstSolution, &AOEST::getTerminateOnFirstSolution, "0,1");
    Planner::declareParam<double>("max_distance", this, &AOEST::setMaxDistance, &AOEST::getMaxDistance, "0.:.1:100");
}

ompl::control::AOEST::~AOEST()
{
    freeMemory();
}

void ompl::control::AOEST::setup()
{
    base::Planner::setup();
    if (!nn_)
        nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    nn_->setDistanceFunction([this](const Motion *a, const Motion *b)
                             {
                                 return distanceFunction(a, b);
                             });

    // === EST: Initialize projection evaluator if not set ===
    if (!projectionEvaluator_)
    {
        projectionEvaluator_ = si_->getStateSpace()->getDefaultProjection();
        if (!projectionEvaluator_)
        {
            OMPL_WARN("%s: No projection evaluator set. EST functionality will be limited.", getName().c_str());
        }
        else
        {
            // Initialize grid with proper dimension
            tree_.grid = Grid<MotionInfo>(projectionEvaluator_->getDimension());
        }
    }

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



void ompl::control::AOEST::clear()
{
    Planner::clear();
    sampler_.reset();
    controlSampler_.reset();
    freeMemory();
    if (nn_)
        nn_->clear();
    if (opt_)
        prevSolutionCost_ = opt_->infiniteCost();
    
    // === EST: Clear grid-based structures ===
    tree_.grid.clear();
    tree_.size = 0;
    pdf_.clear();
    
    // Clear all stored solutions
    allSolutions_.clear();
}

void ompl::control::AOEST::freeMemory()
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

ompl::control::AOEST::Motion *ompl::control::AOEST::selectNode(ompl::control::AOEST::Motion *sample)
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


// === AOEST: Modified solve() to perform EST-style selection and expansion ===
ompl::base::PlannerStatus ompl::control::AOEST::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    base::Goal *goal = pdef_->getGoal().get();

    // Initialize convergence tracking (commented out for production)
    // static std::vector<std::pair<unsigned, double>> convergenceHistory;
    // static unsigned lastConvergenceOutput = 0;
    // static auto lastTime = std::chrono::high_resolution_clock::now();

    while (const base::State *st = pis_.nextStart())
    {
        auto *motion = new Motion(siC_);
        si_->copyState(motion->state_, st);
        siC_->nullControl(motion->control_);
        motion->accCost_ = opt_->identityCost();
        // === EST: Use addMotion for proper grid-based density tracking ===
        addMotion(motion);
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
        // === EST: Use grid-based density tracking for node selection ===
        Motion *nmotion = selectMotion();
        if (!nmotion)
        {
            // Nothing to expand (all inactive)
            break;
        }

        // === AOEST: random control propagation only (no steering toward a sampled state) ===
        controlSampler_->sample(rctrl);
        unsigned int cd = rng_.uniformInt(siC_->getMinControlDuration(), siC_->getMaxControlDuration());
        unsigned int propCd = siC_->propagateWhileValid(nmotion->state_, rctrl, cd, rstate);

        if (propCd == cd)
        {
            base::Cost incCostMotion = opt_->motionCost(nmotion->state_, rstate);
            base::Cost incCostControl = opt_->controlCost(rctrl, cd);
            base::Cost incCost = opt_->combineCosts(incCostMotion, incCostControl);
            base::Cost cost = opt_->combineCosts(nmotion->accCost_, incCost);

            // === AOEST: keep the upper-bound pruning you already have ===
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

                // === EST: Use addMotion for proper grid-based density tracking ===
                addMotion(motion);
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
                    // ...

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

                // === AOEST: removal path unchanged; numChildren_ will naturally keep the density bias in sync ===
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
        
        // Periodic progress and tree statistics (commented out for production)
        // ...
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

    // === EST: Enhanced statistics with grid information ===
    if (projectionEvaluator_)
    {
        OMPL_INFORM("%s: Grid structure: %u cells, %u tree states", getName().c_str(), tree_.grid.size(), tree_.size);
    }

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


ompl::base::PlannerStatus ompl::control::AOEST::resolve(const double replanning_time)
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
    
    // (Optional) preserve original path motions...
    
    OMPL_INFORM("Finished: Marking motions to keep (branch from second state)");
    
    OMPL_INFORM("Replan: Marked %d motions to keep (including original solution path)", keepCount);
    
    // 6. Efficiently rebuild tree with only motions to keep (single rebuild instead of multiple)
    auto startTotal = std::chrono::high_resolution_clock::now();

    // Collect motions to keep and remove
    auto startCollect = std::chrono::high_resolution_clock::now();
    std::vector<Motion*> toKeep;
    std::vector<Motion*> toRemove;
    toKeep.reserve(keepCount);
    toRemove.reserve(allMotions.size() - keepCount);

    for (Motion* motion : allMotions)
    {
        if (motion->toKeep_)
        {
            toKeep.push_back(motion);
        }
        else
        {
            toRemove.push_back(motion);
        }
    }
    auto endCollect = std::chrono::high_resolution_clock::now();
    auto collectTime = std::chrono::duration_cast<std::chrono::milliseconds>(endCollect - startCollect).count();

    // Rebuild tree efficiently with single reconstruction
    auto startRebuild = std::chrono::high_resolution_clock::now();
    nn_->clear();
    if (!toKeep.empty())
    {
        nn_->add(toKeep);
    }
    auto endRebuild = std::chrono::high_resolution_clock::now();
    auto rebuildTime = std::chrono::duration_cast<std::chrono::milliseconds>(endRebuild - startRebuild).count();

    // Rebuild grid structures after tree reconstruction
    auto startGrid = std::chrono::high_resolution_clock::now();
    if (projectionEvaluator_)
    {
        tree_.grid.clear();
        tree_.size = 0;
        pdf_.clear();
        
        // Rebuild grid with kept motions
        for (Motion* motion : toKeep)
        {
            if (motion && !motion->inactive_)
            {
                Grid<MotionInfo>::Coord coord(projectionEvaluator_->getDimension());
                projectionEvaluator_->computeCoordinates(motion->state_, coord);
                Grid<MotionInfo>::Cell* cell = tree_.grid.getCell(coord);
                
                if (cell)
                {
                    cell->data.push_back(motion);
                    if (cell->data.elem_)
                        pdf_.update(cell->data.elem_, 1.0 / cell->data.size());
                }
                else
                {
                    cell = tree_.grid.createCell(coord);
                    cell->data.push_back(motion);
                    tree_.grid.add(cell);
                    cell->data.elem_ = pdf_.add(cell, 1.0);
                }
                tree_.size++;
            }
        }
    }
    auto endGrid = std::chrono::high_resolution_clock::now();
    auto gridTime = std::chrono::duration_cast<std::chrono::milliseconds>(endGrid - startGrid).count();

    // Free memory for removed motions
    auto startCleanup = std::chrono::high_resolution_clock::now();
    int removedCount = 0;
    for (Motion* motion : toRemove)
    {
        if (motion->state_) si_->freeState(motion->state_);
        if (motion->control_) siC_->freeControl(motion->control_);
        delete motion;
        removedCount++;
    }
    auto endCleanup = std::chrono::high_resolution_clock::now();
    auto cleanupTime = std::chrono::duration_cast<std::chrono::milliseconds>(endCleanup - startCleanup).count();

    auto endTotal = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTotal - startTotal).count();
    
    OMPL_INFORM("Tree rebuild timing: collect=%ld ms, rebuild=%ld ms, grid=%ld ms, cleanup=%ld ms, total=%ld ms", 
                collectTime, rebuildTime, gridTime, cleanupTime, totalTime);
    OMPL_INFORM("Replan: Removed %d motions, kept %d motions", removedCount, keepCount);
    
    // 7. Make newStart the root and update problem definition
    newStart->parent_ = nullptr;
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
    pdef_->clearStartStates();
    pdef_->addStartState(newState);
    
    // 8. Create a shortened path from the current solution by removing the first state/control
    auto shortenedPath = std::make_shared<PathControl>(si_);
    
    for (size_t i = 1; i < pathControl->getStateCount(); ++i)
    {
        if (i == 1)
        {
            shortenedPath->append(pathControl->getState(i));
        }
        else if (i < pathControl->getStateCount())
        {
            shortenedPath->append(pathControl->getState(i), 
                               pathControl->getControl(i-1),
                               pathControl->getControlDuration(i-1));
        }
    }
    
    // 9. Add shortened path and continue planning
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
    pdef_->clearSolutionPaths();
    ompl::base::PlannerSolution path2solution(shortenedPath);
    path2solution.cost_ = shortenedCost;
    pdef_->addSolutionPath(path2solution);
    allSolutions_.push_back(path2solution);
    
    OMPL_INFORM("Replan: Created shortened path with %d states, continuing planning for %.2f seconds", 
                shortenedPath->getStateCount(), replanning_time);

    // 10. Run the solve function to continue planning from the new start state
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
    
    OMPL_INFORM("Replan: completed with status: %s", status ? "SUCCESS" : "FAILED");
    return status;
}

void ompl::control::AOEST::getPlannerData(base::PlannerData &data) const
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

void ompl::control::AOEST::costTrackingThread(const std::string& filename, 
                                               std::chrono::time_point<std::chrono::system_clock> startTime) const
{
    // Dummy implementation for Python binding compatibility
    // This function was removed but Python bindings still expect it
    // Do nothing - cost tracking functionality has been removed
    
    // Suppress unused parameter warnings
    (void)filename;
    (void)startTime;
}

const std::vector<ompl::base::PlannerSolution>& ompl::control::AOEST::getAllSolutions() const
{
    return allSolutions_;
}

void ompl::control::AOEST::clearAllSolutions()
{
    allSolutions_.clear();
}

// === EST: Grid-based density tracking methods ===

void ompl::control::AOEST::addMotion(Motion* motion)
{
    // Add to nearest neighbors structure (keep existing functionality)
    nn_->add(motion);
    
    // Add to grid-based density tracking structure
    if (projectionEvaluator_)
    {
        Grid<MotionInfo>::Coord coord(projectionEvaluator_->getDimension());
        projectionEvaluator_->computeCoordinates(motion->state_, coord);
        Grid<MotionInfo>::Cell* cell = tree_.grid.getCell(coord);
        
        if (cell)
        {
            cell->data.push_back(motion);
            if (cell->data.elem_)
                pdf_.update(cell->data.elem_, 1.0 / cell->data.size());
        }
        else
        {
            cell = tree_.grid.createCell(coord);
            cell->data.push_back(motion);
            tree_.grid.add(cell);
            cell->data.elem_ = pdf_.add(cell, 1.0);
        }
        tree_.size++;
    }
}

ompl::control::AOEST::Motion* ompl::control::AOEST::selectMotion()
{
    // Use grid-based density tracking if available
    if (projectionEvaluator_ && !pdf_.empty())
    {
        Grid<MotionInfo>::Cell* cell = pdf_.sample(rng_.uniform01());
        if (cell && !cell->data.empty())
        {
            return cell->data[rng_.uniformInt(0, cell->data.size() - 1)];
        }
    }
    
    // Fallback to your existing selection method for backward compatibility
    std::vector<Motion*> candidates;
    nn_->list(candidates);
    double totalW = 0.0;
    for (auto *m : candidates)
    {
        if (m && !m->inactive_)
            totalW += 1.0 / (1.0 + static_cast<double>(m->numChildren_));
    }
    if (totalW == 0.0)
        return nullptr;
    
    double r = rng_.uniform01() * totalW;
    double acc = 0.0;
    for (auto *m : candidates)
    {
        if (m && !m->inactive_)
        {
            acc += 1.0 / (1.0 + static_cast<double>(m->numChildren_));
            if (acc >= r)
                return m;
        }
    }
    return nullptr;
}
