(define (problem move_fork_to_drawer_problem)
  (:domain robot1)
  (:objects
    robot1 - robot
    fork - object
    drawer - object
    counterTop - object
  )
  (:init
    (at robot1 counterTop)
    (at-location fork counterTop)
    (at-location drawer counterTop) ; This line is not necessary but doesn't affect correctness.
  )
  (:goal
    (and
      (at-location fork drawer)
    )
  )
)