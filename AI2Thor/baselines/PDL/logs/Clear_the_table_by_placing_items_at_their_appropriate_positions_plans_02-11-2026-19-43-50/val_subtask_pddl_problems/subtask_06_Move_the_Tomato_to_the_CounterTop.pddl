(define (problem move-tomato-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    tomato - object
    diningtable - object
    countertop - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 diningtable)
    (at-location tomato diningtable)
    (not (holding robot1 tomato))
  )

  (:goal (and
    (at-location tomato countertop)
    (not (holding robot1 tomato))
  ))
)