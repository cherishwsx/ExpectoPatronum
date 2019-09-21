---
description: Easy
---

# 597. Friend Requests I: Overall Acceptance Rate

See original page [here](https://leetcode.com/problems/friend-requests-i-overall-acceptance-rate/).

## Problem

In social network like Facebook or Twitter, people send friend requests and accept others’ requests as well. Now given two tables as below:

 Table: `friend_request`

```text
| sender_id | send_to_id |request_date|
|-----------|------------|------------|
| 1         | 2          | 2016_06-01 |
| 1         | 3          | 2016_06-01 |
| 1         | 4          | 2016_06-01 |
| 2         | 3          | 2016_06-02 |
| 3         | 4          | 2016-06-09 |
```

 Table: `request_accepted`

```text
| requester_id | accepter_id |accept_date |
|--------------|-------------|------------|
| 1            | 2           | 2016_06-03 |
| 1            | 3           | 2016-06-08 |
| 2            | 3           | 2016-06-08 |
| 3            | 4           | 2016-06-09 |
| 3            | 4           | 2016-06-10 |
```

 Write a query to find the overall acceptance rate of requests rounded to 2 decimals, which is the number of acceptance divide the number of requests.

 For the sample data above, your query should return the following result.

```text
|accept_rate|
|-----------|
|       0.80|
```

 **Note:**

* The accepted requests are not necessarily from the table `friend_request`. In this case, you just need to simply count the total accepted requests \(no matter whether they are in the original requests\), and divide it by the number of requests to get the acceptance rate.
* It is possible that a sender sends multiple requests to the same receiver, and a request could be accepted more than once. In this case, the ‘duplicated’ requests or acceptances are only counted once.
* If there is no requests at all, you should return 0.00 as the accept\_rate.

 **Explanation:** There are 4 unique accepted requests, and there are 5 requests in total. So the rate is 0.80.

 **Follow-up:**

* Can you write a query to return the accept rate but for every month?
* How about the cumulative accept rate for every day?

## Solution

### Details

1. 我对这道题 很无语，follow-up看[这里](https://leetcode.com/problems/friend-requests-i-overall-acceptance-rate/discuss/358575/Detailed-Explaination-for-Question-and-2-follow-up)。
2. The assumption says the acceptance does not need to match the request table. So answer below does not work in Leetcode but it's not wrong in reality.

```sql
SELECT 
     IFNULL(ROUND(COUNT(distinct r.requester_id, r.accepter_id)/COUNT(distinct f.sender_id, f.send_to_id), 2), 0) AS accept_rate
FROM 
     friend_request AS f LEFT JOIN request_accepted AS r
     ON f.sender_id = r.requester_id AND f.send_to_id = r.accepter_id;
```

### Answer

```sql
SELECT ROUND(IFNULL(
       (SELECT COUNT(DISTINCT requester_id, accepter_id) FROM request_accepted) /
       (SELECT COUNT(DISTINCT sender_id, send_to_id) FROM friend_request)
       , 0), 2) AS accept_rate;
```



