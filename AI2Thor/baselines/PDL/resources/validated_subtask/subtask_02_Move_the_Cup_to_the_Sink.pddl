(define (problem move-cup-to-sink)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    cup - object
    sink - object
    floor - object
    kitchen - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 kitchen)
    (at-location cup kitchen)
    (at-location sink floor)
    (not (holding robot1 cup))
  )

  (:goal (and
    (at-location cup sink)
    (not (holding robot1 cup))
  ))

  (:metric minimize (total-cost))
)