(define (problem extinguish-canyonfire)
  (:domain sar_domain)

  (:objects
    robot1 - robot
    ReservoirSierra - object
    CanyonFire_Region_1 - object
    Sand - object
    CanyonFire - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 ReservoirSierra)
    (is-reservoir ReservoirSierra)
    (is-region CanyonFire_Region_1)
    (is-supply Sand)
    (is-fire CanyonFire)
    (fire-active CanyonFire_Region_1)
    (region-of CanyonFire_Region_1 CanyonFire)
    (supply-for-fire Sand CanyonFire)
    (has-resource ReservoirSierra Sand)
  )

  (:goal (and
    (not (fire-active CanyonFire_Region_1))
  ))

  (:metric minimize (total-cost))
)