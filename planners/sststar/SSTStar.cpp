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
#include "ompl/control/planners/sststar/SSTStar.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/base/objectives/MinimaxObjective.h"
#include "ompl/base/objectives/MaximizeMinClearanceObjective.h"
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"
#include "ompl/base/objectives/MechanicalWorkOptimizationObjective.h"

ompl::control::SSTStar::SSTStar(const SpaceInformationPtr &si) : base::Planner(si, "SSTStar")
{
    specs_.approximateSolutions = true;
    siC_ = si.get();
    prevSolution_.clear();
    prevSolutionControls_.clear();
    prevSolutionSteps_.clear();

    Planner::declareParam<double>("goal_bias", this, &SSTStar::setGoalBias, &SSTStar::getGoalBias, "0.:.05:1.");
    Planner::declareParam<double>("selection_radius", this, &SSTStar::setSelectionRadius, &SSTStar::getSelectionRadius, "0.:.1:"
                                                                                                                "100");
    Planner::declareParam<double>("pruning_radius", this, &SSTStar::setPruningRadius, &SSTStar::getPruningRadius, "0.:.1:100");
    Planner::declareParam<bool>("terminate_on_first_solution", this, &SSTStar::setTerminateOnFirstSolution, &SSTStar::getTerminateOnFirstSolution, "0,1");

    // bestSolutionPath_ = std::make_shared<PathControl>(si);
}

ompl::control::SSTStar::~SSTStar()
{
    freeMemory();
}

void ompl::control::SSTStar::setup()
{
    base::Planner::setup();
    if (!nn_)
        nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    nn_->setDistanceFunction([this](const Motion *a, const Motion *b)
                             {
                                 return distanceFunction(a, b);
                             });
    if (!witnesses_)
        witnesses_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    witnesses_->setDistanceFunction([this](const Motion *a, const Motion *b)
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



void ompl::control::SSTStar::clear()
{
    Planner::clear();
    sampler_.reset();
    controlSampler_.reset();
    freeMemory();
    if (nn_)
        nn_->clear();
    if (witnesses_)
        witnesses_->clear();
    if (opt_)
        prevSolutionCost_ = opt_->infiniteCost();
    
    // Clear all stored solutions
    allSolutions_.clear();

}

void ompl::control::SSTStar::freeMemory()
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
    if (witnesses_)
    {
        std::vector<Motion *> witnesses;
        witnesses_->list(witnesses);
        for (auto &witness : witnesses)
        {
            delete witness;
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

ompl::control::SSTStar::Motion *ompl::control::SSTStar::selectNode(ompl::control::SSTStar::Motion *sample)
{
    std::vector<Motion *> ret;
    Motion *selected = nullptr;
    base::Cost bestCost = opt_->infiniteCost();
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
        int k = 1;
        while (selected == nullptr)
        {
            nn_->nearestK(sample, k, ret);
            for (unsigned int i = 0; i < ret.size() && selected == nullptr; i++)
                if (!ret[i]->inactive_)
                    selected = ret[i];
            k += 5;
        }
    }
    return selected;
}

ompl::control::SSTStar::Witness *ompl::control::SSTStar::findClosestWitness(ompl::control::SSTStar::Motion *node)
{
    if (witnesses_->size() > 0)
    {
        auto *closest = static_cast<Witness *>(witnesses_->nearest(node));
        if (distanceFunction(closest, node) > pruningRadius_)
        {
            closest = new Witness(siC_);
            closest->linkRep(node);
            si_->copyState(closest->state_, node->state_);
            witnesses_->add(closest);
        }
        return closest;
    }
    else
    {
        auto *closest = new Witness(siC_);
        closest->linkRep(node);
        si_->copyState(closest->state_, node->state_);
        witnesses_->add(closest);
        return closest;
    }
}

ompl::base::PlannerStatus ompl::control::SSTStar::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    base::Goal *goal = pdef_->getGoal().get();
    auto *goal_s = dynamic_cast<base::GoalSampleableRegion *>(goal);

    while (const base::State *st = pis_.nextStart())
    {
        auto *motion = new Motion(siC_);
        si_->copyState(motion->state_, st);
        siC_->nullControl(motion->control_);
        nn_->add(motion);
        motion->accCost_ = opt_->identityCost();
        findClosestWitness(motion);
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
            Witness *closestWitness = findClosestWitness(rmotion);

            if (closestWitness->rep_ == rmotion || opt_->isCostBetterThan(cost, closestWitness->rep_->accCost_))
            {
                Motion *oldRep = closestWitness->rep_;
                /* create a motion */
                auto *motion = new Motion(siC_);
                motion->accCost_ = cost;
                si_->copyState(motion->state_, rmotion->state_);
                siC_->copyControl(motion->control_, rctrl);
                motion->steps_ = cd;
                motion->parent_ = nmotion;
                nmotion->children_.push_back(motion);
                nmotion->numChildren_++;
                closestWitness->linkRep(motion);

                nn_->add(motion);
                double dist = 0.0;
                bool solv = goal->isSatisfied(motion->state_, &dist);
                            
                
                if (solv)
                {
                    approxdif = dist;
                    solution = motion;

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



                    OMPL_INFORM("Found solution with cost %.4f", solution->accCost_.value());
                    // TODO: Update bestSolutionCost_ and bestSolutionPath_

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

                if (oldRep != rmotion)
                {
                    while (oldRep->inactive_ && oldRep->numChildren_ == 0)
                    {
                        oldRep->inactive_ = true;
                        nn_->remove(oldRep);

                        if (oldRep->state_)
                            si_->freeState(oldRep->state_);
                        if (oldRep->control_)
                            siC_->freeControl(oldRep->control_);

                        oldRep->state_ = nullptr;
                        oldRep->control_ = nullptr;
                        oldRep->parent_->numChildren_--;
                        oldRep->parent_->children_.erase(std::remove(oldRep->parent_->children_.begin(), oldRep->parent_->children_.end(), oldRep), oldRep->parent_->children_.end());
                        Motion *oldRepParent = oldRep->parent_;
                        delete oldRep;
                        oldRep = oldRepParent;
                    }
                }
            }
        }
        iterations++;
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
        /* set the solution path */
        // auto path(std::make_shared<PathControl>(si_));
        // for (int i = prevSolution_.size() - 1; i >= 1; --i)
        //     path->append(prevSolution_[i], prevSolutionControls_[i - 1],
        //                  prevSolutionSteps_[i - 1] * siC_->getPropagationStepSize());
        // path->append(prevSolution_[0]);
        solved = true;
        // pdef_->addSolutionPath(path, approximate, approxdif, getName());
    }

    si_->freeState(xstate);
    if (rmotion->state_)
        si_->freeState(rmotion->state_);
    if (rmotion->control_)
        siC_->freeControl(rmotion->control_);
    delete rmotion;



    OMPL_INFORM("%s: Created %u states in %u iterations", getName().c_str(), nn_->size(), iterations);

    return {solved, approximate};
}

ompl::base::PlannerStatus ompl::control::SSTStar::resolve(const double replanning_time)
{
    checkValidity();
    const auto resolveStarted = std::chrono::steady_clock::now();
    const bool profileResolve = std::getenv("AURA_SSTSTAR_PROFILE") != nullptr;
    auto profileMark = resolveStarted;
    auto profileLap = [&](const char *label) {
        if (!profileResolve) return;
        const auto now = std::chrono::steady_clock::now();
        OMPL_INFORM("%s: [profile] %s took %.6f s", getName().c_str(), label,
                    std::chrono::duration<double>(now - profileMark).count());
        profileMark = now;
    };
    
    // Ensure the planner has a tree to work with
    if (!nn_ || nn_->size() == 0)
    {
        OMPL_WARN("%s: No tree to resolve.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    // Get the current solution path and SAVE IT before calling solve()
    ompl::base::PathPtr path = pdef_->getSolutionPath();
    auto pathControl = std::dynamic_pointer_cast<PathControl>(path);

    if (!pathControl || pathControl->getStateCount() < 2)
    {
        OMPL_WARN("%s: No valid solution path with at least 2 states available for resolve.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    // SAVE the original path states before solve() overwrites them
    std::vector<ompl::base::State*> originalPathStates;
    for (size_t i = 0; i < pathControl->getStateCount(); ++i) {
        ompl::base::State* stateCopy = si_->allocState();
        si_->copyState(stateCopy, pathControl->getState(i));
        originalPathStates.push_back(stateCopy);
    }
    
    // Find the motion in the tree that corresponds to the second state in the solution path.
    // This will become our new root/start motion.
    // Note: We want to replan from the second state, so we need to find the motion
    // that corresponds to the second state in the original path
    ompl::base::State *secondStateInPath = originalPathStates[1];  // Use saved state
    
    Motion *tempMotionForSearch = new Motion(siC_);
    si_->copyState(tempMotionForSearch->state_, secondStateInPath);
    Motion *newStartMotion = nn_->nearest(tempMotionForSearch);
    si_->freeState(tempMotionForSearch->state_);
    siC_->freeControl(tempMotionForSearch->control_);
    delete tempMotionForSearch;

    if (!newStartMotion)
    {
        for (base::State* state : originalPathStates)
            si_->freeState(state);
        OMPL_WARN("%s: Could not find nearest motion to second state.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    double distance = si_->distance(newStartMotion->state_, secondStateInPath);
    if (distance > 1e-5)
    {
        for (base::State* state : originalPathStates)
            si_->freeState(state);
        OMPL_WARN("%s: Could not find the second state of the solution path in the tree (distance=%.8f).", getName().c_str(), distance);
        return base::PlannerStatus::ABORT;
    }
    profileLap("save path + locate new root");

    // Remove everything except the subtree rooted at newStartMotion
    // Collect all motions that should be KEPT (newStartMotion and its descendants)
    std::set<Motion*> motionsToKeep;
    std::queue<Motion*> subtreeQueue;
    subtreeQueue.push(newStartMotion);
    motionsToKeep.insert(newStartMotion);
    
    while (!subtreeQueue.empty()) {
        Motion* current = subtreeQueue.front();
        subtreeQueue.pop();
        
        // Add all children to keep set and queue
        for (Motion* child : current->children_) {
            motionsToKeep.insert(child);
            subtreeQueue.push(child);
        }
    }
    
    // Get all motions currently in the tree
    std::vector<Motion*> allMotions;
    nn_->list(allMotions);
    
    // Rebuild once instead of repeatedly removing nodes from the nearest-
    // neighbor structure. The latter is quadratic for large SSTStar trees.
    std::vector<Motion*> keptMotions;
    std::vector<Motion*> removedMotions;
    keptMotions.reserve(motionsToKeep.size());
    removedMotions.reserve(allMotions.size() - motionsToKeep.size());
    for (Motion* motion : allMotions) {
        (motionsToKeep.find(motion) == motionsToKeep.end()
             ? removedMotions
             : keptMotions)
            .push_back(motion);
    }
    nn_->clear();
    if (!keptMotions.empty())
        nn_->add(keptMotions);
    for (Motion* motion : removedMotions) {
        if (motion->state_) si_->freeState(motion->state_);
        if (motion->control_) siC_->freeControl(motion->control_);
        delete motion;
    }
    if (profileResolve)
        OMPL_INFORM("%s: [profile] tree sizes: kept=%zu removed=%zu", getName().c_str(),
                    keptMotions.size(), removedMotions.size());
    profileLap("BFS subtree selection + nn_ clear/add + free removed motions");

    // Make newStartMotion the new root (no parent) but KEEP its children
    newStartMotion->parent_ = nullptr;
    std::queue<Motion*> costQueue;
    costQueue.push(newStartMotion);
    newStartMotion->accCost_ = opt_->identityCost();
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
    profileLap("re-root + accCost_ BFS recompute");

    // Rebuild witness set: witnesses whose representative motion survived
    // pruning above are still exactly valid (pruning only removes
    // candidates -- a surviving rep_ is still the best known motion in its
    // neighborhood among the survivors), so keep those with one bulk
    // re-add instead of deleting and recreating every witness. Ordinary
    // (non-representative) surviving motions don't need eager coverage
    // restored here: the witness set's only job is gating which *new*
    // candidates get accepted during growth (see the accCost_ comparison
    // in solve()), not tracking every existing tree node, and any
    // surviving motion left without an active witness right now gets one
    // (or is superseded by a better nearby candidate) the same lazy way
    // any freshly-sampled point normally does. This does not touch nn_,
    // parent_/children_, or accCost_ -- the kinodynamic tree's
    // parent-child structure and every motion's cost are unaffected.
    //
    // Previously this rebuilt from scratch by calling findClosestWitness()
    // once per surviving motion -- O(|kept tree|) nearest-neighbor
    // operations on every resolve() call, which for small pruning radii
    // (where nearly every motion becomes its own witness) made this
    // rebuild consume most or all of a short replanning budget.
    if (witnesses_)
    {
        std::vector<Motion*> existing_witnesses;
        witnesses_->list(existing_witnesses);

        std::vector<Motion*> keptWitnesses;
        keptWitnesses.reserve(existing_witnesses.size());
        for (Motion* w : existing_witnesses)
        {
            auto *witness = static_cast<Witness*>(w);
            if (motionsToKeep.find(witness->rep_) != motionsToKeep.end())
                keptWitnesses.push_back(w);
            else
                delete witness;
        }
        witnesses_->clear();
        if (!keptWitnesses.empty())
            witnesses_->add(keptWitnesses);
    }
    profileLap("witness rebuild");

    // Set the new start state in the problem definition
    pdef_->clearStartStates();
    pdef_->addStartState(newStartMotion->state_);
    
    // Ensure the goal is properly set (it should still be there, but make sure)
    ompl::base::GoalPtr goal = pdef_->getGoal();
    if (!goal) {
        for (base::State* state : originalPathStates)
            si_->freeState(state);
        OMPL_ERROR("%s: No goal set in problem definition", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    // Preserve the exact unexecuted suffix before continuing the search. This
    // keeps the ProblemDefinition consistent with its new start even if no
    // better path is found inside the online budget.
    auto shortenedPath = std::make_shared<PathControl>(si_);
    for (std::size_t i = 1; i < pathControl->getStateCount(); ++i)
    {
        if (i == 1)
            shortenedPath->append(originalPathStates[i]);
        else
            shortenedPath->append(
                originalPathStates[i],
                pathControl->getControl(i - 1),
                pathControl->getControlDuration(i - 1));
    }
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
    pdef_->clearSolutionPaths();
    base::PlannerSolution shortenedSolution(shortenedPath);
    shortenedSolution.cost_ = shortenedCost;
    pdef_->addSolutionPath(shortenedSolution);
    allSolutions_.clear();
    allSolutions_.push_back(shortenedSolution);
    prevSolutionCost_ = shortenedCost;
    bestSolutionCost_ = shortenedCost;

    profileLap("shortened-path bookkeeping");

    // Call solve to replan from new start.
    pdef_->setGoal(goal);
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
        OMPL_WARN("%s: Tree maintenance consumed the %.6f s replanning budget; retaining the current exact path.",
                  getName().c_str(), replanning_time);

    // The solve() call has created a new solution path starting from newStartMotion
    // This new path should be the complete solution (starting from the new start state)
    if (status)
    {
        ompl::base::PathPtr newPath = pdef_->getSolutionPath();
        auto newPathControl = std::dynamic_pointer_cast<PathControl>(newPath);
        
        if (newPathControl && newPathControl->getStateCount() > 0)
        {
            // The new path from solve() already starts from newStartMotion and goes to the goal
            // This is exactly what we want - a complete solution from the new start state
            // No need to modify it further, just ensure it's properly set
            
            OMPL_INFORM("Resolve successful: new solution path created starting from new start state");
        }
    }
    else
    {
        OMPL_WARN("Resolve failed: could not find new solution from replanning");
    }

    // Clean up saved original path states
    for (ompl::base::State* state : originalPathStates) {
        si_->freeState(state);
    }

    return status;
}

// ompl::base::PlannerStatus ompl::control::SSTStar::simple_resolve(const double replanning_time)
// {
//     checkValidity();
    
//     // 1. Get the current states and control in the solution path
//     ompl::base::PathPtr path = pdef_->getSolutionPath();
//     auto pathControl = std::dynamic_pointer_cast<PathControl>(path);

//     if (!pathControl || pathControl->getStateCount() < 2)
//     {
//         OMPL_WARN("%s: No valid solution path with at least 2 states available for simple_resolve.", getName().c_str());
//         return base::PlannerStatus::ABORT;
//     }
    
//     // Debug: Print original path info
//     OMPL_INFORM("Simple_resolve: Original path has %d states and %d controls", 
//                 pathControl->getStateCount(), pathControl->getControlCount());
//     if (pathControl->getStateCount() > 0)
//     {
//         ompl::base::State* firstState = pathControl->getState(0);
//         double x = firstState->as<ompl::base::SE2StateSpace::StateType>()->getX();
//         double y = firstState->as<ompl::base::SE2StateSpace::StateType>()->getY();
//         double yaw = firstState->as<ompl::base::SE2StateSpace::StateType>()->getYaw();
//         OMPL_INFORM("Simple_resolve: Original path starts at (%.3f, %.3f, %.3f)", x, y, yaw);
   
//     // 2. Change the start state to the next state in solution path
//     ompl::base::State *nextState = pathControl->getState(1);
//     pdef_->clearStartStates();
//     pdef_->addStartState(nextState);
    
//     // 3. Run the solve function for the given replanning_time
//     base::PlannerTerminationCondition ptc = base::timedPlannerTerminationCondition(replanning_time);
//     base::PlannerStatus status = solve(ptc);
    
//     // 4. If solve succeeded, use its result (which should start from the new start state)
//     // If solve failed, create a fallback path by removing the first state/control
//     if (status)
//     {
//         // solve() succeeded and created a new solution path starting from the new start state
//         // This is exactly what we want - a replanned solution from the current position
//         OMPL_INFORM("Simple_resolve: solve() succeeded, using replanned solution");
        
//         // Debug: Check what the new solution path looks like
//         ompl::base::PathPtr newPath = pdef_->getSolutionPath();
//         auto newPathControl = std::dynamic_pointer_cast<PathControl>(newPath);
//         if (newPathControl && newPathControl->getStateCount() > 0)
//         {
//             ompl::base::State* firstState = newPathControl->getState(0);
//             double x = firstState->as<ompl::base::SE2StateSpace::StateType>()->getX();
//             double y = firstState->as<ompl::base::SE2StateSpace::StateType>()->getY();
//             double yaw = firstState->as<ompl::base::SE2StateSpace::StateType>()->getYaw();
//             OMPL_INFORM("Simple_resolve: New solution starts at (%.3f, %.3f, %.3f)", x, y, yaw);
            
//             // Check if it matches the expected start state
//             double expectedX = nextState->as<ompl::base::SE2StateSpace::StateType>()->getX();
//             double expectedY = nextState->as<ompl::base::SE2StateSpace::StateType>()->getY();
//             double expectedYaw = nextState->as<ompl::base::SE2StateSpace::StateType>()->getYaw();
//             OMPL_INFORM("Simple_resolve: Expected start at (%.3f, %.3f, %.3f)", expectedX, expectedY, expectedYaw);
            
//             if (std::abs(x - expectedX) > 0.01 || std::abs(y - expectedY) > 0.01 || std::abs(yaw - expectedYaw) > 0.01)
//             {
//                 OMPL_WARN("Simple_resolve: New solution doesn't start from expected position!");
//             }
//         }
        
//         return status;
//     }
//     else
//     {
//         // solve() failed, create a fallback path by removing the first state/control
//         OMPL_INFORM("Simple_resolve: solve() failed, using fallback path (removing first state/control)");
        
//         auto fallbackPath = std::make_shared<PathControl>(si_);
        
//         // Add all states and controls starting from index 1 (skip the first state)
//         for (size_t i = 1; i < pathControl->getStateCount(); ++i)
//         {
//             if (i == 1)
//             {
//                 // First state in fallback path (was second state in original path)
//                 fallbackPath->append(pathControl->getState(i));
//             }
//             else
//             {
//                 // Add control and state for remaining segments
//                 fallbackPath->append(pathControl->getState(i), 
//                                    pathControl->getControl(i-1),
//                                    pathControl->getControlDuration(i-1));
//             }
//         }
        
//         pdef_->clearSolutionPaths();
//         pdef_->addSolutionPath(fallbackPath, true, 0.0, getName());
//         return base::PlannerStatus(true, true); // solved, approximate
//     }
// }

ompl::base::PlannerStatus ompl::control::SSTStar::replan(const double replanning_time)
{
    checkValidity();
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
    
    auto pathControl = std::dynamic_pointer_cast<PathControl>(bestSolutionIt->path_);
    OMPL_INFORM("Using best solution from allSolutions_ with cost %.4f and %d states", 
                bestSolutionIt->cost_.value(), 
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
    
    // 6. Rebuild the nearest-neighbor structure once with the marked subtree.
    // Repeated nn_->remove() calls are prohibitively expensive for the large
    // trees produced by the initial solve.
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
    
    // 7. Clear witnesses and rebuild
    OMPL_INFORM("Starting: Clear and rebuild witnesses");
    if (witnesses_)
    {
        std::vector<Motion*> existing_witnesses;
        witnesses_->list(existing_witnesses);
        for(auto& w : existing_witnesses) {
            delete w;
        }
        witnesses_->clear();
    }
    OMPL_INFORM("Finished: Clear and rebuild witnesses");
    
    // 8. Make newStart the root (no parent)
    OMPL_INFORM("Starting: Make newStart the root");
    newStart->parent_ = nullptr;
    
    // 9. Recalculate costs for all remaining motions
    // std::queue<Motion*> costQueue;
    // costQueue.push(newStart);
    // newStart->accCost_ = opt_->identityCost(); // Root has identity cost
    
    // while (!costQueue.empty())
    // {
    //     Motion* current = costQueue.front();
    //     costQueue.pop();
        
    //     for (Motion* child : current->children_)
    //     {
    //         // Calculate incremental cost from parent to child
    //         base::Cost incCostMotion = opt_->motionCost(current->state_, child->state_);
    //         base::Cost incCostControl = opt_->controlCost(child->control_, child->steps_);
    //         base::Cost incCost = opt_->combineCosts(incCostMotion, incCostControl);
            
    //         // Set child's accumulated cost
    //         child->accCost_ = opt_->combineCosts(current->accCost_, incCost);
            
    //         costQueue.push(child);
    //     }
    // }
    
    // 10. Rebuild witness set for remaining motions
    OMPL_INFORM("Starting: Rebuild witness set for remaining motions");
    std::vector<Motion*> remaining_motions;
    nn_->list(remaining_motions);
    for (Motion* m : remaining_motions) {
        findClosestWitness(m);
    }
    OMPL_INFORM("Finished: Rebuild witness set for remaining motions");
    
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
    
    // Recompute the suffix cost. PathControl::cost() only supports geometric
    // paths in upstream OMPL, and retaining the pre-trim cost corrupts the
    // comparison against newly discovered solutions.
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
    base::PlannerTerminationCondition ptc = base::timedPlannerTerminationCondition(replanning_time);
    base::PlannerStatus status = solve(ptc);
    OMPL_INFORM("Finished: Run solve to continue planning from new start state");
    
    OMPL_INFORM("Replan: solve() completed with status: %s", status ? "SUCCESS" : "FAILED");
    OMPL_INFORM("Finished: replan function");
    return status;
}

void ompl::control::SSTStar::getPlannerData(base::PlannerData &data) const
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

void ompl::control::SSTStar::costTrackingThread(const std::string& filename, 
                                               std::chrono::time_point<std::chrono::system_clock> startTime) const
{
    // Dummy implementation for Python binding compatibility
    // This function was removed but Python bindings still expect it
    // Do nothing - cost tracking functionality has been removed
}

const std::vector<ompl::base::PlannerSolution>& ompl::control::SSTStar::getAllSolutions() const
{
    return allSolutions_;
}

void ompl::control::SSTStar::clearAllSolutions()
{
    allSolutions_.clear();
}
