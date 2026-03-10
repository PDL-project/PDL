(define (problem moveforktostorage_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    fork - object
    drawer - object
    countertop - object
  )
  (:init
    (at robot1 countertop)
    (at-location fork countertop)
    (not (holding robot1 fork))
  )
  (:goal
    (and
      (at-location fork drawer)
      (not (holding robot1 fork))
    )
  )
)