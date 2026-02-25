```lisp
(define (problem extinguish-plainfire)
  (:domain sar_domain)

  (:objects
    robot1 - robot
    ReservoirDelta - object
    PlainFire_Region_1 - object
    Water - object
    PlainFire - object
  )

  (:init
    (at robot1 kitchen)
    (is-reservoir ReservoirDelta)
    (is-region PlainFire_Region_1)
    (is-supply Water)
    (fire-active PlainFire_Region_1)
    (region-of PlainFire_Region_1 PlainFire)
    (supply-for-fire Water PlainFire)
    (has-resource ReservoirDelta Water)
  )

  (:goal (and
    (not (fire-active PlainFire_Region_1))
  ))
)
```