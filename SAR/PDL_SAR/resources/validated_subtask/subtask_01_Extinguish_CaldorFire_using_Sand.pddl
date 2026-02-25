(define (problem extinguish-caldorfire)
  (:domain sar_domain)

  (:objects
    robot1 - robot
    ReservoirUtah - object
    CaldorFire_Region_1 - object
    CaldorFire_Region_2 - object
    Sand - object
    CaldorFire - object
  )

  (:init
    (= (total-cost) 0)
    (at robot1 ReservoirUtah)
    (is-reservoir ReservoirUtah)
    (is-region CaldorFire_Region_1)
    (is-region CaldorFire_Region_2)
    (is-supply Sand)
    (fire-active CaldorFire_Region_1)
    (fire-active CaldorFire_Region_2)
    (region-of CaldorFire_Region_1 CaldorFire)
    (region-of CaldorFire_Region_2 CaldorFire)
    (supply-for-fire Sand CaldorFire)
    (has-resource ReservoirUtah Sand)
  )

  (:goal (and
    (not (fire-active CaldorFire_Region_1))
    (not (fire-active CaldorFire_Region_2))
  ))

  (:metric minimize (total-cost))
)