```pddl
(define (problem extinguish-ridgefire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    ReservoirSierra - object
    RidgeFire_Region_1 - object
    Sand - object
    RidgeFire - object
  )

  (:init
    (at robot1 kitchen)
    (is-reservoir ReservoirSierra)
    (is-region RidgeFire_Region_1)
    (is-supply Sand)
    (fire-active RidgeFire_Region_1)
    (region-of RidgeFire_Region_1 RidgeFire)
    (supply-for-fire Sand RidgeFire)
    (has-resource ReservoirSierra Sand)
  )

  (:goal (and
    (not (fire-active RidgeFire_Region_1))
  ))
)
```