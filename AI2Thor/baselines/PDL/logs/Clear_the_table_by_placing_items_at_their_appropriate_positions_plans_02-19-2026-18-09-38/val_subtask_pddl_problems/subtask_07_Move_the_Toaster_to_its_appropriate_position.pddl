(define (problem move-toaster-to-appropriate-position)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    toaster - object
    diningtable - object
    countertop - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location toaster diningtable)
    (not (holding robot1 toaster))
  )

  (:goal (and
    (at-location toaster countertop)
    (not (holding robot1 toaster))
  ))

  (:metric minimize (total-cost))
)