(define (problem extinguish-harborfire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    ReservoirDelta - object
    HarborFire_Region_1 - object
    Water - object
    HarborFire - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 ReservoirDelta)
    (is-reservoir ReservoirDelta)
    (is-region HarborFire_Region_1)
    (is-supply Water)
    (fire-active HarborFire_Region_1)
    (region-of HarborFire_Region_1 HarborFire)
    (supply-for-fire Water HarborFire)
    (has-resource ReservoirDelta Water)
  )

  (:goal (and
    (not (fire-active HarborFire_Region_1))
  ))

  (:metric minimize (total-cost))
)