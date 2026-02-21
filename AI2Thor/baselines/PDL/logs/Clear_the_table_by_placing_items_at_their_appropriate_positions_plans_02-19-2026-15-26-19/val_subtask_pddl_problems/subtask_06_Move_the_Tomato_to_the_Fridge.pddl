(define (problem move-tomato-to-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato - object
    fridge - object
    diningtable - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location tomato diningtable)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
<<<<<<<< Updated upstream:AI2Thor/baselines/PDL/logs/Clear_the_table_by_placing_items_at_their_appropriate_positions_plans_02-19-2026-15-26-19/val_subtask_pddl_problems/subtask_06_Move_the_Tomato_to_the_Fridge.pddl
========
    (object-close robot1 fridge)
>>>>>>>> Stashed changes:AI2Thor/baselines/PDL/logs/Put_the_bread,_lettuce,_and_tomato_in_the_fridge_plans_02-21-2026-14-20-31/val_subtask_pddl_problems/subtask_03_Put_the_Tomato_in_the_Fridge.pddl
    (not (holding robot1 tomato))
  )

  (:goal (and
    (at-location tomato fridge)
    (not (holding robot1 tomato))
    (object-close robot1 fridge)
  ))

  (:metric minimize (total-cost))
)