```pddl
(define (problem extinguish-greatfire)
  (:domain sar_domain)

  (:objects
    robot1 - robot
    ReservoirYork - object
    GreatFire_Region_1 - object
    Water - object
  )

  (:init
    (at robot1 kitchen)
    (is-reservoir ReservoirYork)
    (is-region GreatFire_Region_1)
    (is-supply Water)
    (has-resource ReservoirYork Water)
    (fire-active GreatFire_Region_1)
    (region-of GreatFire_Region_1 GreatFire)
    (supply-for-fire Water GreatFire)
  )

  (:goal (and
    (not (fire-active GreatFire_Region_1))
  ))
)
```