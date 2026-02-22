(define (problem move-coffeemachine)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    coffeemachine - object
    diningtable - object
    countertop - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location coffeemachine diningtable)
    (at-location diningtable floor)
    (at-location countertop floor)
    (not (holding robot1 coffeemachine))
  )

  (:goal (and
    (at-location coffeemachine countertop)
    (not (holding robot1 coffeemachine))
  ))
)