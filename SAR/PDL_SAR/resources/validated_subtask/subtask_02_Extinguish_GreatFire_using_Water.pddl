(define (problem extinguish-greatfire)
  (:domain sar_domain)

  (:objects
    robot1 - robot
    ReservoirYork - object
    GreatFire_Region_1 - object
    Water - object
    GreatFire - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 kitchen)
    (is-reservoir ReservoirYork)
    (is-region GreatFire_Region_1)
    (is-supply Water)
    (is-fire GreatFire)
    (has-resource ReservoirYork Water)
    (fire-active GreatFire_Region_1)
    (region-of GreatFire_Region_1 GreatFire)
    (supply-for-fire Water GreatFire)
  )

  (:goal (and
    (not (fire-active GreatFire_Region_1))
  ))

  (:metric minimize (total-cost))
)