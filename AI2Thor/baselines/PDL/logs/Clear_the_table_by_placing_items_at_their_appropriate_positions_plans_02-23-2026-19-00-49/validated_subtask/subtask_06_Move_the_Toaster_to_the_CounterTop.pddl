(define (problem move-toaster-to-countertop)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    toaster - object
    diningTable - object
    counterTop - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location toaster diningTable)
    (at-location counterTop floor)
    (not (holding robot1 toaster))
    (not (object-close robot1 counterTop)) ;; Ensure counterTop is not closed
  )

  (:goal (and
    (at-location toaster counterTop)
    (not (holding robot1 toaster))
  ))

  (:metric minimize (total-cost))
)