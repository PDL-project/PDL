(define (problem put-egg-in-fridge)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    egg - object
    fridge - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (at-location egg floor)
    (at-location fridge floor)

    (is-fridge fridge)
    (not (fridge-open fridge))
    (object-close robot1 fridge)
    (not (holding robot1 egg))
  )

  (:goal (and
    (at-location egg fridge)
    (not (holding robot1 egg))
    (object-close robot1 fridge)
  ))

  (:metric minimize (total-cost))
)