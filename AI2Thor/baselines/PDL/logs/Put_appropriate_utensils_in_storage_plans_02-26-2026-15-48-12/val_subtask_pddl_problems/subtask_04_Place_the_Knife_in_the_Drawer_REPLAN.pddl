(define (problem place-knife-in-drawer)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    drawer1 - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location knife floor)
    (at-location drawer1 floor)
    (object-close robot1 drawer1)
  )

  (:goal (and
    (at-location knife drawer1)
    (not (holding robot1 knife))
    (object-close robot1 drawer1)
  ))
)