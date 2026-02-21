(define (problem move-bread-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread - object
    countertop - object
    diningtable - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
<<<<<<<< Updated upstream:AI2Thor/baselines/PDL/logs/Clear_the_table_by_placing_items_at_their_appropriate_positions_plans_02-19-2026-15-12-19/val_subtask_pddl_problems/subtask_02_Move_the_Bread_to_the_CounterTop.pddl
    (at robot1 diningtable)
    (at-location bread diningtable)
    (at-location countertop floor)
========
    (at robot1 countertop)
    (at-location bread countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (object-close robot1 fridge)
>>>>>>>> Stashed changes:AI2Thor/baselines/PDL/logs/Put_the_bread,_lettuce,_and_tomato_in_the_fridge_plans_02-21-2026-14-20-31/val_subtask_pddl_problems/subtask_01_Put_the_Bread_in_the_Fridge.pddl
    (not (holding robot1 bread))
  )

  (:goal (and
    (at-location bread countertop)
    (not (holding robot1 bread))
  ))

  (:metric minimize (total-cost))
)