(define (problem move_book_to_shelf_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    book - object
    shelf - object
    couch - object
    floor - object
  )
  (:init
    (at robot1 floor)
    (at-location book couch)
  )
  (:goal
    (and
      (at-location book shelf)
    )
  )
)