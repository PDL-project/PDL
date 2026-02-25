(define (problem extinguish-moorfire)
  (:domain sar_domain)

  (:objects
    robot1 - robot
    ReservoirSierra - object
    MoorFire_Region_1 - object
    Sand - object
    MoorFire - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 kitchen)
    (is-reservoir ReservoirSierra)
    (is-region MoorFire_Region_1)
    (is-supply Sand)
    (is-fire MoorFire)
    (fire-active MoorFire_Region_1)
    (region-of MoorFire_Region_1 MoorFire)
    (supply-for-fire Sand MoorFire)
    (has-resource ReservoirSierra Sand)
  )

  (:goal (and
    (not (fire-active MoorFire_Region_1))
  ))

  (:metric minimize (total-cost))
)