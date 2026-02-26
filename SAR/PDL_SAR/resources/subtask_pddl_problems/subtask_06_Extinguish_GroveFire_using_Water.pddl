```lisp
(define (problem extinguish-grovefire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    ReservoirDelta - object
    GroveFire_Region_1 - object
    Water - object
    GroveFire - object
    kitchen - object
  )

  (:init
    (at robot1 kitchen)
    (is-reservoir ReservoirDelta)
    (is-region GroveFire_Region_1)
    (is-supply Water)
    (fire-active GroveFire_Region_1)
    (region-of GroveFire_Region_1 GroveFire)
    (supply-for-fire Water GroveFire)
    (has-resource ReservoirDelta Water)
  )

  (:goal (and
    (not (fire-active GroveFire_Region_1))
  ))
)
```