(define (problem open-fridge1)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    fridge1 - object
    floor - object
    kitchen - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location fridge1 floor)

    (is-fridge fridge1)
    (not (fridge-open fridge1))
    (object-close robot1 fridge1)
  )

  (:goal (and
    (object-open robot1 fridge1)
    (fridge-open fridge1)
  ))
)