```pddl
(define (problem move-fork-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    fork - object
    diningtable - object
    countertop - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location fork diningtable)
    (at-location diningtable floor)
    (at-location countertop floor)

    (not (holding robot1 fork))
  )

  (:goal (and
    (at-location fork countertop)
    (not (holding robot1 fork))
  ))
)
```