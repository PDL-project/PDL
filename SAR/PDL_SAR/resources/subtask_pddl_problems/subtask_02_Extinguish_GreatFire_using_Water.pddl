```lisp
(define (problem extinguish-greatfire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    base_camp - object
    ReservoirYork - object
    GreatFire_Region_1 - object
    Water - object
    GreatFire - object
  )

  (:init
    (at robot1 base_camp)
    (is-supply Water)
    (has-resource ReservoirYork Water)
    (fire-active GreatFire_Region_1)
    (is-region GreatFire_Region_1)
    (region-of GreatFire_Region_1 GreatFire)
    (supply-for-fire Water GreatFire)
  )

  (:goal (and
    (not (fire-active GreatFire_Region_1))
  ))
)
```