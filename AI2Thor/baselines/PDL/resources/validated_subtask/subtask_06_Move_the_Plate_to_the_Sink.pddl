(define (problem move-plate-to-sink)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    plate - object
    sink - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location plate kitchen)
    (at-location sink floor)
    (not (holding robot1 plate))
  )

  (:goal (and
    (at-location plate sink)
    (not (holding robot1 plate))
  ))

  (:metric minimize (total-cost))
)