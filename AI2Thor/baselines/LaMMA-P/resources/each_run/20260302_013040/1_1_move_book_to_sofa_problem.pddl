(define (problem move_book_to_sofa_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    book - object
    sideTable - object
    sofa - object
  )
  (:init
    (at robot1 sideTable)
    (at-location book sideTable)
  )
  (:goal
    (and
      (at-location book sofa)
    )
  )
)