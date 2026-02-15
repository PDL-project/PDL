(define (problem move-tomato-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato - object
    diningTable - object
    counterTop - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 diningTable)
    (at-location tomato diningTable)
    (not (holding robot1 tomato))
  )

  (:goal (and
    (at-location tomato counterTop)
    (not (holding robot1 tomato))
  ))
)