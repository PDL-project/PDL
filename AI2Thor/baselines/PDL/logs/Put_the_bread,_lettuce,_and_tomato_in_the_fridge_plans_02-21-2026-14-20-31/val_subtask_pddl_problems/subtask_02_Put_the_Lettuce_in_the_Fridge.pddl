(define (problem move-tomato-to-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato - object
    fridge - object
    diningtable - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location tomato diningtable)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 tomato))
  )

  (:goal (and
<<<<<<<< Updated upstream:AI2Thor/baselines/PDL/logs/Clear_the_table_by_placing_items_at_their_appropriate_positions_plans_02-19-2026-15-12-19/val_subtask_pddl_problems/subtask_06_Move_the_Tomato_to_the_Fridge.pddl
    (at-location tomato fridge)
    (not (holding robot1 tomato))
========
    (at-location lettuce fridge)
    (not (holding robot1 lettuce))
>>>>>>>> Stashed changes:AI2Thor/baselines/PDL/logs/Put_the_bread,_lettuce,_and_tomato_in_the_fridge_plans_02-21-2026-14-20-31/val_subtask_pddl_problems/subtask_02_Put_the_Lettuce_in_the_Fridge.pddl
    (object-close robot1 fridge)
  ))

  (:metric minimize (total-cost))
)