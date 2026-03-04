```lisp
(define (problem extinguish-plainfire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    base_camp - object
    ReservoirDelta - object
    PlainFire_Region_1 - object
    Water - object
    PlainFire - object
  )

  (:init
    (at robot1 base_camp)
    (is-supply Water)
    (has-resource ReservoirDelta Water)
    (fire-active PlainFire_Region_1)
    (is-region PlainFire_Region_1)
    (region-of PlainFire_Region_1 PlainFire)
    (supply-for-fire Water PlainFire)
  )

  (:goal (and
    (not (fire-active PlainFire_Region_1))
  ))
)
```