(define (problem extinguish-harborfire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    base_camp - object
    ReservoirDelta - object
    HarborFire_Region_1 - object
    Water - object
    HarborFire - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 base_camp)
    (is-supply Water)
    (has-resource ReservoirDelta Water)
    (fire-active HarborFire_Region_1)
    (is-region HarborFire_Region_1)
    (region-of HarborFire_Region_1 HarborFire)
    (supply-for-fire Water HarborFire)
    (is-fire HarborFire)
    (is-reservoir ReservoirDelta)
  )

  (:goal (and
    (not (fire-active HarborFire_Region_1))
  ))

  (:metric minimize (total-cost))
)