```
(define (problem extinguish-grovefire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    base_camp - object
    ReservoirDelta - object
    GroveFire_Region_1 - object
    Water - object
    GroveFire - object
  )

  (:init
    ; Robot starts at base camp — GoToObject will navigate it to the supply source.
    (at robot1 base_camp)
    (is-supply Water)
    (has-resource ReservoirDelta Water)
    (fire-active GroveFire_Region_1)
    (is-region GroveFire_Region_1)
    (region-of GroveFire_Region_1 GroveFire)
    (supply-for-fire Water GroveFire)
  )

  (:goal (and
    (not (fire-active GroveFire_Region_1))
  ))
)
```