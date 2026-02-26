(define (problem placeholder-task)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    knife - object
    countertop - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location knife countertop)
    (holding robot1 knife)
  )

  (:goal (and
    ;; No specific goal condition is defined for this placeholder task.
  ))

  (:metric minimize (total-cost))
)