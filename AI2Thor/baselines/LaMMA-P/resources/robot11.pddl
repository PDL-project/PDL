(define (domain robot11)
  (:requirements :strips :typing :negative-preconditions :adl :quantified-preconditions :conditional-effects)
  (:types robot object)
  (:predicates
    (at ?robot - robot ?object - object)
    (inaction ?robot - robot)
    (holding ?robot - robot ?object - object)
    (at-location ?object - object ?location - object)
    (switch-on ?robot - robot ?object - object)
    (switch-off ?robot - robot ?object - object)
    (object-open ?robot - robot ?object - object)
    (object-close ?robot - robot ?object - object)
    (break ?robot - robot ?object - object)
    (sliced ?object - object)
    (cleaned ?robot - robot ?object - object)
    (is-fridge ?object - object)
  )

  (:action GoToObject
    :parameters (?robot - robot ?object - object)
    :precondition (not (inaction ?robot))
    :effect (and
      (at ?robot ?object)
      (forall (?another_object - object)
        (when (at ?robot ?another_object)
          (not (at ?robot ?another_object))
        )
      )
      (not (inaction ?robot))
    )
  )

  (:action PickupObject
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and
      (at-location ?object ?location)
      (at ?robot ?location)
      (not (inaction ?robot))
    )
    :effect (and
      (holding ?robot ?object)
      (not (inaction ?robot))
    )
  )

  (:action PutObject
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and
      (holding ?robot ?object)
      (not (inaction ?robot))
      (at ?robot ?location)
      (or (not (is-fridge ?location))
          (object-open ?robot ?location))
    )
    :effect (and
      (at-location ?object ?location)
      (not (holding ?robot ?object))
      (not (inaction ?robot))
    )
  )

  (:action SwitchOn
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (switch-on ?robot ?object)
    )
  )

  (:action Switchoff
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (switch-off ?robot ?object)
    )
  )

  (:action OpenObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (object-open ?robot ?object)
      (not (object-close ?robot ?object))
    )
  )

  (:action BreakObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (break ?robot ?object)
    )
  )

  (:action CloseObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (object-close ?robot ?object)
      (not (object-open ?robot ?object))
    )
  )

  (:action SliceObject
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and
      (at-location ?object ?location)
      (at ?robot ?location)
      (not (inaction ?robot))
    )
    :effect (and
      (not (inaction ?robot))
      (sliced ?object)
    )
  )

  (:action CleanObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (cleaned ?robot ?object)
    )
  )

  (:action OpenFridge
    :parameters (?robot - robot ?fridge - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?fridge)
      (is-fridge ?fridge)
    )
    :effect (and
      (not (inaction ?robot))
      (object-open ?robot ?fridge)
      (not (object-close ?robot ?fridge))
    )
  )

  (:action CloseFridge
    :parameters (?robot - robot ?fridge - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?fridge)
      (object-open ?robot ?fridge)
      (is-fridge ?fridge)
    )
    :effect (and
      (not (inaction ?robot))
      (object-close ?robot ?fridge)
      (not (object-open ?robot ?fridge))
    )
  )
)
