(define (problem extinguish-moorfire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    base_camp - object
    ReservoirSierra - object
    MoorFire_Region_1 - object
    Sand - object
    MoorFire - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 base_camp)
    (is-supply Sand)
    (has-resource ReservoirSierra Sand)
    (fire-active MoorFire_Region_1)
    (is-region MoorFire_Region_1)
    (region-of MoorFire_Region_1 MoorFire)
    (supply-for-fire Sand MoorFire)
    (is-reservoir ReservoirSierra)
    (is-fire MoorFire)
  )

  (:goal (and
    (not (fire-active MoorFire_Region_1))
  ))

  (:metric minimize (total-cost))
)