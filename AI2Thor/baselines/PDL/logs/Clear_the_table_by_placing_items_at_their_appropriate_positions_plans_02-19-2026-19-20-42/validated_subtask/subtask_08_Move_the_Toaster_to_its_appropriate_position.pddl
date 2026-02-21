(define (problem move-toaster-to-appropriate-position)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    toaster - object
    diningTable - object
    counterTop - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location toaster diningTable)
    (not (holding robot1 toaster))
  )

  (:goal (and
    (at-location toaster counterTop)
    (not (holding robot1 toaster))
  ))

  (:metric minimize (total-cost))
)