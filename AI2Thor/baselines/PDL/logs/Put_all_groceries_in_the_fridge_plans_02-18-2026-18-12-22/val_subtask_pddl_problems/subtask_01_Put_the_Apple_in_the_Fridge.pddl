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
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location apple countertop)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (object-close robot1 fridge)
    (not (holding robot1 apple))
  )

  (:goal (and
    (at-location apple fridge)
    (not (holding robot1 apple))
    (object-close robot1 fridge)
  ))

  (:metric minimize (total-cost))
)