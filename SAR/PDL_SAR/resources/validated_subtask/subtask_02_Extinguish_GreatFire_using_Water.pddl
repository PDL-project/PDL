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
    (= (total-cost) 0)
    (at robot1 base_camp)
    (is-supply Water)
    (has-resource ReservoirYork Water)
    (fire-active GreatFire_Region_1)
    (is-region GreatFire_Region_1)
    (region-of GreatFire_Region_1 GreatFire)
    (supply-for-fire Water GreatFire)
    (is-fire GreatFire)
    (is-reservoir ReservoirYork)
  )

  (:goal (and
    (not (fire-active GreatFire_Region_1))
  ))

  (:metric minimize (total-cost))
)