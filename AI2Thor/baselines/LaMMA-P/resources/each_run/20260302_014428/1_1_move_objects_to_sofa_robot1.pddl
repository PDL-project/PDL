(define (problem move_objects_to_sofa_robot1)
  (:domain robot1)
  (:objects
    robot1 - robot
    book - object
    bowl - object
    plate - object
    diningTable - object
    sofa - object
  )
  (:init
    (at robot1 diningTable)
    (at-location book diningTable)
    (at-location bowl diningTable)
    (at-location plate diningTable)
  )
  (:goal
    (and
      (at-location book sofa)
      (at-location bowl sofa)
      (at-location plate sofa)
      (not (holding robot1 book))
      (not (holding robot1 bowl))
      (not (holding robot1 plate))
    )
  )
)