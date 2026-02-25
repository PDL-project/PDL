(define (problem put-apple-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    apple - object
    fridge - object
    kitchen - object
    countertop - object
    floor - object
  )

  (:init
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location apple countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (not (holding robot1 apple))
    (object-close robot1 fridge)
  )

  (:goal (and
    (at-location apple fridge)
    (object-close robot1 fridge)
  ))
)