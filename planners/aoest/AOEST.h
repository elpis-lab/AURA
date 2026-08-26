/*
 * Author: Ali Golestaneh
 */


#ifndef OMPL_CONTROL_PLANNERS_AO_EST_
#define OMPL_CONTROL_PLANNERS_AO_EST_

#include <chrono>
#include <limits>
#include <string>
#include <vector>
#include "ompl/control/planners/PlannerIncludes.h"
#include "ompl/datastructures/NearestNeighbors.h"
#include "ompl/datastructures/Grid.h"
#include "ompl/datastructures/PDF.h"
#include "ompl/base/ProjectionEvaluator.h"

namespace ompl
{
    namespace control
    {
        /**
           @anchor cAOEST
           @par Short description
           \ref cAOEST "AOEST" (Asymptotically Optimal EST) is a asymptotically near-optimal incremental
           sampling-based motion planning algorithm for systems with dynamics. It makes use
           of random control inputs to perform a search for the best control inputs to explore
           the state space.
           @par External documentation
           Ali Golestaneh, Kostas E. Bekris, Asymptotically Optimal EST.
           [[PDF]](https://arxiv.org/abs/2508.12345)
        */
        class AOEST : public base::Planner
        {
        public:
            /** \brief Constructor */
            AOEST(const SpaceInformationPtr &si);

            ~AOEST() override;

            void setup() override;

            /** \brief Continue solving for some amount of time. Return true if solution was found. */
            base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc) override;

            /** \brief Continue solving for some amount of time. Return true if solution was found. */
            base::PlannerStatus resolve(const double replanning_time);

            /** \brief Get the cost of the best solution found so far. */
            base::Cost getBestSolutionCost() const
            {
                return prevSolutionCost_;
            }


            void getPlannerData(base::PlannerData &data) const override;

            /** \brief Dummy cost tracking thread function for Python binding compatibility */
            void costTrackingThread(const std::string& filename, 
                                   std::chrono::time_point<std::chrono::system_clock> startTime) const;

            /** \brief Get all solutions found during planning */
            const std::vector<ompl::base::PlannerSolution>& getAllSolutions() const;

            /** \brief Clear all stored solutions */
            void clearAllSolutions();

            /** \brief Clear datastructures. Call this function if the
                input data to the planner has changed and you do not
                want to continue planning */
            void clear() override;

            /** In the process of randomly selecting states in the state
                space to attempt to go towards, the algorithm may in fact
                choose the actual goal state, if it knows it, with some
                probability. This probability is a real number between 0.0
                and 1.0; its value should usually be around 0.05 and
                should not be too large. It is probably a good idea to use
                the default value. */
            void setGoalBias(double goalBias)
            {
                goalBias_ = goalBias;
                dGoalBias_ = goalBias;
            }

            /** \brief Get the goal bias the planner is using */
            double getGoalBias() const
            {
                return goalBias_;
            }

            /**
                \brief Set the radius for selecting nodes relative to random sample.

                This radius is used to mimic behavior of RRT* in that it promotes
                extending from nodes with good path cost from the root of the tree.
                Making this radius larger will provide higher quality paths, but has two
                major drawbacks; exploration will occur much more slowly and exploration
                around the boundary of the state space may become impossible. */
            void setSelectionRadius(double selectionRadius)
            {
                selectionRadius_ = selectionRadius;
            }

            /** \brief Get the selection radius the planner is using */
            double getSelectionRadius() const
            {
                return selectionRadius_;
            }


            /** \brief When set to true, the planner will terminate the first time
                it finds a solution. Otherwise, it will continue to search for
                a better solution until the planner termination condition is met. */
            void setTerminateOnFirstSolution(bool terminate)
            {
                terminateOnFirstSolution_ = terminate;
            }

            /** \brief Get whether the planner is configured to terminate on the
                first solution. */
            bool getTerminateOnFirstSolution() const
            {
                return terminateOnFirstSolution_;
            }

            /** \brief Set a different nearest neighbors datastructure */
            template <template <typename T> class NN>
            void setNearestNeighbors()
            {
                if (nn_ && nn_->size() != 0)
                    OMPL_WARN("Calling setNearestNeighbors will clear all states.");
                clear();
                nn_ = std::make_shared<NN<Motion *>>();
                setup();
            }

        protected:
            /** \brief Representation of a motion

                This only contains pointers to parent motions as we
                only need to go backwards in the tree. */
            class Motion
            {
            public:
                Motion() = default;

                /** \brief Constructor that allocates memory for the state and the control */
                Motion(const SpaceInformation *si)
                  : state_(si->allocState()), control_(si->allocControl())
                {
                }

                virtual ~Motion() = default;

                virtual base::State *getState() const
                {
                    return state_;
                }
                virtual Motion *getParent() const
                {
                    return parent_;
                }

                base::Cost accCost_{0};

                /** \brief The state contained by the motion */
                base::State *state_{nullptr};

                /** \brief The control contained by the motion */
                Control *control_{nullptr};

                /** \brief The number of steps_ the control is applied for */
                unsigned int steps_{0};

                /** \brief The parent motion in the exploration tree */
                Motion *parent_{nullptr};

                /** \brief The list of children motions in the exploration tree */
                std::vector<Motion *> children_;

                /** \brief Number of children */
                unsigned numChildren_{0};

                /** \brief If inactive, this node is not considered for selection.*/
                bool inactive_{false};

                /** \brief Flag to keep this motion during replanning.*/
                bool toKeep_{false};
            };

            /** \brief Finds the best node in the tree withing the selection radius around a random sample.*/
            Motion *selectNode(Motion *sample);

            /** \brief Free the memory allocated by this planner */
            void freeMemory();

            /** \brief Compute distance between motions (actually distance between contained states) */
            double distanceFunction(const Motion *a, const Motion *b) const
            {
                return si_->distance(a->state_, b->state_);
            }

            /** \brief Set the projection evaluator for grid-based density tracking */
            void setProjectionEvaluator(const base::ProjectionEvaluatorPtr &projectionEvaluator)
            {
                projectionEvaluator_ = projectionEvaluator;
            }

            /** \brief Get the projection evaluator */
            const base::ProjectionEvaluatorPtr& getProjectionEvaluator() const
            {
                return projectionEvaluator_;
            }

            /** \brief Set the maximum distance for motion extension */
            void setMaxDistance(double maxDistance)
            {
                maxDistance_ = maxDistance;
            }

            /** \brief Get the maximum distance for motion extension */
            double getMaxDistance() const
            {
                return maxDistance_;
            }

            /** \brief State sampler */
            base::StateSamplerPtr sampler_;

            /** \brief Control sampler */
            ControlSamplerPtr controlSampler_;

            /** \brief The base::SpaceInformation cast as control::SpaceInformation, for convenience */
            const SpaceInformation *siC_;

            /** \brief A nearest-neighbors datastructure containing the tree of motions */
            std::shared_ptr<NearestNeighbors<Motion *>> nn_;

            /** \brief The fraction of time the goal is picked as the state to expand towards (if such a state is
             * available) */
            double goalBias_{0.05};
            double dGoalBias_{0.05};

            /** \brief The radius for determining the node selected for extension. */
            double selectionRadius_{0.5};

            /** \brief Flag indicating whether to terminate planning when the first solution is found. */
            bool terminateOnFirstSolution_{false};

            /** \brief The random number generator */
            RNG rng_;

            /** \brief The best solution we found so far. */
            std::vector<base::State *> prevSolution_;
            std::vector<Control *> prevSolutionControls_;
            std::vector<unsigned> prevSolutionSteps_;

            /** \brief The best solution cost we found so far. */
            base::Cost prevSolutionCost_;

            /** \brief The best cost found so far as a double value. */
            ompl::base::Cost bestSolutionCost_{std::numeric_limits<double>::infinity()};
            std::shared_ptr<PathControl> bestSolutionPath_;

            /** \brief Vector to store all planner solutions found during planning */
            std::vector<ompl::base::PlannerSolution> allSolutions_;

            /** \brief The optimization objective. */
            base::OptimizationObjectivePtr opt_;

            // === EST: Grid-based density tracking structures ===
            /** \brief Motion info for grid-based density tracking */
            struct MotionInfo
            {
                Motion *operator[](unsigned int i)
                {
                    return motions_[i];
                }
                const Motion *operator[](unsigned int i) const
                {
                    return motions_[i];
                }
                void push_back(Motion *m)
                {
                    motions_.push_back(m);
                }
                unsigned int size() const
                {
                    return motions_.size();
                }
                bool empty() const
                {
                    return motions_.empty();
                }
                std::vector<Motion *> motions_;
                PDF<Grid<MotionInfo>::Cell*>::Element *elem_{nullptr};
            };

            /** \brief A grid cell */
            using GridCell = Grid<MotionInfo>::Cell;

            /** \brief A PDF of grid cells */
            using CellPDF = PDF<GridCell*>;

            /** \brief The data contained by a tree of exploration */
            struct TreeData
            {
                TreeData() = default;

                /** \brief A grid where each cell contains an array of motions */
                Grid<MotionInfo> grid{0};

                /** \brief The total number of motions in the grid */
                unsigned int size{0};
            };

            /** \brief Grid-based exploration tree */
            TreeData tree_;

            /** \brief PDF for selecting grid cells based on density */
            CellPDF pdf_;

            /** \brief Projection evaluator for state space projection */
            base::ProjectionEvaluatorPtr projectionEvaluator_;

            /** \brief Maximum distance for motion extension */
            double maxDistance_{0.0};

            // === EST: Helper methods ===
            /** \brief Add a motion to the grid-based tree structure */
            void addMotion(Motion* motion);

            /** \brief Select a motion using grid-based density tracking */
            Motion* selectMotion();

        };
    }
}

#endif
