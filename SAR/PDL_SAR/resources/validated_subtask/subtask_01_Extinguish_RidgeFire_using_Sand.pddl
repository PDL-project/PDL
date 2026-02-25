(define (problem extinguish-ridgefire)
  (:domain sar_domain)

  (:objects
    robot1 - robot
    ReservoirSierra - object
    RidgeFire_Region_1 - object
    Sand - object
    RidgeFire - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 ReservoirSierra)
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

  (:metric minimize (total-cost))
)