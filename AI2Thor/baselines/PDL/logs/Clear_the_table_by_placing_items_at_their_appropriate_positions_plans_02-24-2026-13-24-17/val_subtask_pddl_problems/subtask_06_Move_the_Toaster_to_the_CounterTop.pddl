(define (problem move-toaster-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    toaster - object
    countertop - object
    diningtable - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location toaster diningtable)
    (at-location countertop floor)
    (not (holding robot1 toaster))
  )

  (:goal (and
    (at-location toaster countertop)
    (not (holding robot1 toaster))
  ))

  (:metric minimize (total-cost))
)