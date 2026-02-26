```lisp
(define (problem extinguish-canyonfire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    ReservoirSierra - object
    CanyonFire_Region_1 - object
    Sand - object
    CanyonFire - object
  )

  (:init
    (at robot1 kitchen)
    (fire-active CanyonFire_Region_1)
    (is-region CanyonFire_Region_1)
    (is-reservoir ReservoirSierra)
    (is-supply Sand)
    (has-resource ReservoirSierra Sand)
    (region-of CanyonFire_Region_1 CanyonFire)
    (supply-for-fire Sand CanyonFire)
  )

  (:goal (and
    (not (fire-active CanyonFire_Region_1))
  ))
)
```