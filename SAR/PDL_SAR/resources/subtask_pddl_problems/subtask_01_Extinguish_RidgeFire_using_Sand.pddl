```lisp
(define (problem extinguish-ridgefire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    base_camp - object
    ReservoirSierra - object
    RidgeFire_Region_1 - object
    Sand - object
    RidgeFire - object
  )

  (:init
    (at robot1 base_camp)
    (is-supply Sand)
    (has-resource ReservoirSierra Sand)
    (fire-active RidgeFire_Region_1)
    (is-region RidgeFire_Region_1)
    (region-of RidgeFire_Region_1 RidgeFire)
    (supply-for-fire Sand RidgeFire)
  )

  (:goal (and
    (not (fire-active RidgeFire_Region_1))
  ))
)
```