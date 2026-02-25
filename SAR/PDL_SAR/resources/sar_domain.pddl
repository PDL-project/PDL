(define (domain sar_domain)
  ;; SAR(Search And Rescue) domain for PDL pipeline.
  ;; Actions:
  ;;   GoToObject   -> NavigateTo(name)
  ;;   GetSupply    -> GetSupply(src, supply_type)
  ;;   UseSupply    -> UseSupply(fire, supply_type)
  ;;   Explore      -> Explore()
  ;;   Carry        -> Carry(person)
  ;;   DropOff      -> DropOff(deposit, person)
  ;;   StoreSupply  -> StoreSupply(deposit)

  (:requirements :strips :typing :negative-preconditions :adl :action-costs)

  (:types robot object)

  (:predicates
    ;; robot position
    (at ?r - robot ?o - object)

    ;; supply in robot inventory
    (has-supply ?r - robot ?s - object)

    ;; fire / region active
    (fire-active ?f - object)

    ;; person state
    (person-found ?p - object)
    (carrying ?r - robot ?p - object)
    (person-rescued ?p - object)

    ;; object type flags
    (is-fire ?f - object)
    (is-region ?reg - object)
    (is-reservoir ?res - object)
    (is-deposit ?d - object)
    (is-person ?p - object)
    (is-supply ?s - object)

    ;; region belongs to fire
    (region-of ?reg - object ?fire - object)

    ;; which supply extinguishes which fire
    (supply-for-fire ?s - object ?f - object)

    ;; resource available at source
    (has-resource ?src - object ?s - object)
  )

  (:functions (total-cost) - number)

  ;; ------------------------------------------------------------------
  ;; Navigate to any named object (reservoir, region, deposit, person, fire)
  ;; ------------------------------------------------------------------
  (:action GoToObject
    :parameters (?r - robot ?dest - object)
    :precondition ()
    :effect (and
      (at ?r ?dest)
      (forall (?x - object)
        (when (and (at ?r ?x) (not (= ?x ?dest)))
          (not (at ?r ?x))
        )
      )
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------------
  ;; Collect supply from reservoir or deposit
  ;; ------------------------------------------------------------------
  (:action GetSupply
    :parameters (?r - robot ?src - object ?s - object)
    :precondition (and
      (at ?r ?src)
      (has-resource ?src ?s)
      (is-supply ?s)
    )
    :effect (and
      (has-supply ?r ?s)
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------------
  ;; Use supply on a fire region (robot must be at the fire region)
  ;; ------------------------------------------------------------------
  (:action UseSupply
    :parameters (?r - robot ?reg - object ?s - object)
    :precondition (and
      (at ?r ?reg)
      (has-supply ?r ?s)
      (is-region ?reg)
      (fire-active ?reg)
      (exists (?f - object)
        (and
          (region-of ?reg ?f)
          (supply-for-fire ?s ?f)
        )
      )
    )
    :effect (and
      (not (fire-active ?reg))
      (not (has-supply ?r ?s))
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------------
  ;; Explore to find lost persons
  ;; ------------------------------------------------------------------
  (:action Explore
    :parameters (?r - robot)
    :precondition ()
    :effect (and
      (forall (?p - object)
        (when (is-person ?p)
          (person-found ?p)
        )
      )
      (increase (total-cost) 5)
    )
  )

  ;; ------------------------------------------------------------------
  ;; Carry a found person
  ;; ------------------------------------------------------------------
  (:action Carry
    :parameters (?r - robot ?p - object)
    :precondition (and
      (at ?r ?p)
      (is-person ?p)
      (person-found ?p)
      (not (carrying ?r ?p))
    )
    :effect (and
      (carrying ?r ?p)
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------------
  ;; Drop off carried person at deposit
  ;; ------------------------------------------------------------------
  (:action DropOff
    :parameters (?r - robot ?p - object ?d - object)
    :precondition (and
      (at ?r ?d)
      (carrying ?r ?p)
      (is-deposit ?d)
    )
    :effect (and
      (not (carrying ?r ?p))
      (person-rescued ?p)
      (increase (total-cost) 1)
    )
  )

  ;; ------------------------------------------------------------------
  ;; Store supply at deposit (makes it available for other robots)
  ;; ------------------------------------------------------------------
  (:action StoreSupply
    :parameters (?r - robot ?d - object ?s - object)
    :precondition (and
      (at ?r ?d)
      (is-deposit ?d)
      (has-supply ?r ?s)
    )
    :effect (and
      (not (has-supply ?r ?s))
      (has-resource ?d ?s)
      (increase (total-cost) 1)
    )
  )
)
