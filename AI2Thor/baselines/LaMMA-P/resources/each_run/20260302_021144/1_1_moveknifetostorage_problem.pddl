(define (problem moveknifetostorage_problem)
  (:domain robot2)
  (:objects
    robot2 - robot
    knife - object
    drawer - object
    countertop - object
  )
  (:init
    (at robot2 countertop)
    (at-location knife countertop)
    (not (holding robot2 knife))
  )
  (:goal
    (and 
      (at-location knife drawer)
      (not (holding robot2 knife))
    )
  )
)