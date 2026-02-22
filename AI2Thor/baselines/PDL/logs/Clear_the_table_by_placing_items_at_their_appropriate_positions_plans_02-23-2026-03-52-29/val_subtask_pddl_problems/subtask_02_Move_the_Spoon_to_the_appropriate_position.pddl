(define (problem move-spoon-to-appropriate-position)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spoon - object
    diningtable - object
    countertop - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location spoon diningtable)
    (not (holding robot1 spoon))
    (not (object-close robot1 countertop)) ;; Ensure countertop is not closed
  )

  (:goal (and
    (at-location spoon countertop)
    (not (holding robot1 spoon))
  ))

  (:metric minimize (total-cost))
)