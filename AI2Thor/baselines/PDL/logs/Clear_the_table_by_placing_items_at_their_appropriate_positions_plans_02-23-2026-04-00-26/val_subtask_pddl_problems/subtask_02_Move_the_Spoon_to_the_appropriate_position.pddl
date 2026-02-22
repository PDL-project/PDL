(define (problem move-spoon-to-appropriate-position)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    spoon - object
    diningTable - object
    counterTop - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location spoon diningTable)
    (not (holding robot1 spoon))
    (not (object-close robot1 counterTop))
  )

  (:goal (and
    (at-location spoon counterTop)
    (object-close robot1 counterTop)
  ))

  (:metric minimize (total-cost))
)