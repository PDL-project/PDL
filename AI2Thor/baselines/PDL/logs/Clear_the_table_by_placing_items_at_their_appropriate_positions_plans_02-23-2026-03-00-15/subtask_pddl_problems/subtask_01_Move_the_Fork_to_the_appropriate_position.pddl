```pddl
(define (problem move-fork-to-appropriate-position)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    fork - object
    diningtable - object
    countertop - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location fork diningtable)
    (not (holding robot1 fork))
  )

  (:goal (and
    (at-location fork countertop)
    (not (holding robot1 fork))
  ))
)
```