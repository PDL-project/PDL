```pddl
(define (problem extinguish-caldorfire)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    base_camp - object
    ReservoirUtah - object
    CaldorFire_Region_1 - object
    CaldorFire_Region_2 - object
    Sand - object
    CaldorFire - object
  )

  (:init
    (at robot1 base_camp)
    (is-supply Sand)
    (has-resource ReservoirUtah Sand)
    (fire-active CaldorFire_Region_1)
    (fire-active CaldorFire_Region_2)
    (is-region CaldorFire_Region_1)
    (is-region CaldorFire_Region_2)
    (region-of CaldorFire_Region_1 CaldorFire)
    (region-of CaldorFire_Region_2 CaldorFire)
    (supply-for-fire Sand CaldorFire)
  )

  (:goal (and
    (not (fire-active CaldorFire_Region_1))
    (not (fire-active CaldorFire_Region_2))
  ))
)
```