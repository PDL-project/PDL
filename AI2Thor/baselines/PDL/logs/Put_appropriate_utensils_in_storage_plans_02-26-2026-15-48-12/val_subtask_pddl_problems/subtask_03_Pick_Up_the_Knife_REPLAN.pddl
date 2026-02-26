(define (problem pick-up-knife)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 knife)
    (at-location knife floor)
    (not (holding robot1 knife))
  )

  (:goal (and
    (holding robot1 knife)
  ))
)