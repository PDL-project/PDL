(define (problem move_book_to_sofa_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    book - object
    sofa - object
    sideTable - object
    floor - object
  )
  (:init
    (at robot2 floor)
    (at-location book sideTable)
  )
  (:goal
    (and
      (at-location book sofa)
    )
  )
)